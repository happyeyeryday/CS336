import os
import pickle
import time
import math
import argparse
from contextlib import contextmanager, nullcontext
from typing import List, Dict
from threading import Thread
from queue import Queue

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np

# 导入所有配置和模型组件
from config import data_config, model_config, training_config, inference_config
from cs336_basics.Transformer import TransformerLM
from cs336_basics.bpe import BPE

# --- 流水线数据加载器 ---
class StreamingTokenizer:
    """后台线程异步编码数据的流式tokenizer"""
    def __init__(self, file_path, tokenizer, chunk_size_mb=100):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024
        self.queue = Queue(maxsize=2)  # 最多缓存2个chunk
        self.encoding_thread = None
        self.stop_flag = False

    def start(self):
        """启动后台编码线程"""
        self.encoding_thread = Thread(target=self._encode_worker, daemon=True)
        self.encoding_thread.start()
        print(f"✅ 后台编码线程已启动 (chunk大小: {self.chunk_size_bytes / 1024 / 1024:.1f} MB)")

    def _encode_worker(self):
        """后台线程工作函数：持续读取和编码数据"""
        chunk_id = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            while not self.stop_flag:
                print(f"🔄 [编码线程] 开始编码 chunk {chunk_id}...")
                start_time = time.time()

                tokens_list = []
                bytes_read = 0
                lines_read = 0

                # 读取一个chunk的数据
                while bytes_read < self.chunk_size_bytes:
                    line = f.readline()
                    if not line:  # 文件结束
                        break

                    tokens = self.tokenizer.encode_to_ids(line)
                    tokens_list.extend(tokens)
                    bytes_read += len(line.encode('utf-8'))
                    lines_read += 1

                if not tokens_list:  # 没有数据了，重新开始
                    f.seek(0)
                    chunk_id = 0
                    continue

                # 转换为numpy数组
                tokens_array = np.array(tokens_list, dtype=np.uint16)
                elapsed = time.time() - start_time

                print(f"✅ [编码线程] chunk {chunk_id} 完成: "
                      f"{len(tokens_array):,} tokens, "
                      f"{lines_read:,} 行, "
                      f"用时 {elapsed:.1f}s, "
                      f"速度 {len(tokens_array)/elapsed/1000:.1f}K tokens/s")

                # 放入队列（如果队列满了会阻塞）
                self.queue.put((chunk_id, tokens_array))
                chunk_id += 1

    def get_chunk(self):
        """获取一个编码好的chunk"""
        return self.queue.get()  # 阻塞直到有数据

    def stop(self):
        """停止后台线程"""
        self.stop_flag = True
        if self.encoding_thread:
            self.encoding_thread.join(timeout=5)


class StreamingTextDataset(IterableDataset):
    def __init__(self, streaming_tokenizer, context_length, batch_size, shuffle=True):
        self.streaming_tokenizer = streaming_tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_tokens = None
        self.current_chunk_id = -1
        self.position = 0  # 添加位置追踪

    def _load_next_chunk(self):
        """加载下一个chunk"""
        chunk_id, tokens = self.streaming_tokenizer.get_chunk()
        self.current_chunk_id = chunk_id
        self.current_tokens = tokens
        self.position = 0  # 重置位置
        print(f"📦 [训练进程] 已加载 chunk {chunk_id}: {len(tokens):,} tokens")

    def __iter__(self):
        """定义迭代器逻辑"""
        while True:
            # 检查是否需要加载新chunk
            if self.current_tokens is None:
                self._load_next_chunk()

            # 检查当前chunk是否已经用完（非shuffle模式）
            if not self.shuffle:
                max_start = len(self.current_tokens) - self.context_length - 1
                if self.position + self.batch_size > max_start:
                    print(f"📦 [训练进程] chunk {self.current_chunk_id} 已用完，加载下一个...")
                    self._load_next_chunk()

            # 生成batch
            if self.shuffle:
                # shuffle模式：随机采样
                # 为了充分利用数据，可以设置每个chunk的使用次数
                max_samples_per_chunk = len(self.current_tokens) // self.context_length
                if not hasattr(self, 'chunk_sample_count'):
                    self.chunk_sample_count = 0

                if self.chunk_sample_count >= max_samples_per_chunk:
                    print(f"📦 [训练进程] chunk {self.current_chunk_id} 已充分使用({self.chunk_sample_count}次采样)，加载下一个...")
                    self._load_next_chunk()
                    self.chunk_sample_count = 0

                indices = torch.randint(0, len(self.current_tokens) - self.context_length, (self.batch_size,))
                self.chunk_sample_count += self.batch_size
            else:
                # 顺序模式：连续采样
                indices = torch.arange(self.position, self.position + self.batch_size)
                indices = indices % (len(self.current_tokens) - self.context_length)
                self.position += self.batch_size

            x = torch.stack([torch.from_numpy(self.current_tokens[i:i+self.context_length].astype(np.int64)) for i in indices])
            y = torch.stack([torch.from_numpy(self.current_tokens[i+1:i+self.context_length+1].astype(np.int64)) for i in indices])

            yield x, y


def get_streaming_dataloader(file_path, tokenizer, batch_size, context_length, chunk_size_mb=100, shuffle=True):
    """创建流式DataLoader"""
    streaming_tokenizer = StreamingTokenizer(file_path, tokenizer, chunk_size_mb)
    streaming_tokenizer.start()
    dataset = StreamingTextDataset(streaming_tokenizer, context_length, batch_size, shuffle)
    return DataLoader(dataset, batch_size=None, shuffle=False), streaming_tokenizer


# --- 保留原来的完整缓存版本（用于验证集） ---
class TextDataset(IterableDataset):
    """原来的完整缓存版本，用于较小的验证集"""
    def __init__(self, file_path, tokenizer, context_length, batch_size, shuffle=True):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.shuffle = shuffle

        base_filename = os.path.basename(file_path)
        local_cache_dir = os.path.join(os.path.dirname(__file__), "data", "token_cache2")
        os.makedirs(local_cache_dir, exist_ok=True)
        token_cache_path = os.path.join(local_cache_dir, base_filename + ".tokens.bin")

        if not os.path.exists(token_cache_path):
            print(f"缓存文件 {token_cache_path} 不存在，正在创建...")
            print(f"正在以流式方式加载和分词数据集: {file_path}...")

            with open(token_cache_path, 'wb') as f_cache:
                total_tokens = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        tokens = np.array(self.tokenizer.encode_to_ids(line), dtype=np.uint16)
                        f_cache.write(tokens.tobytes())
                        total_tokens += len(tokens)
                        if (i + 1) % 100000 == 0:
                            print(f"  已处理 {i+1} 行, 总计 {total_tokens} tokens...")
            print(f"分词和缓存创建完毕，总计 {total_tokens} tokens。")

        print(f"正在从缓存文件加载 tokens: {token_cache_path}")
        self.tokens = np.memmap(token_cache_path, dtype=np.uint16, mode='r')
        print(f"数据集加载完毕，总计 {len(self.tokens)} 个 tokens。")

    def __iter__(self):
        if self.shuffle:
            indices = torch.randint(0, len(self.tokens) - self.context_length, (self.batch_size,))
        else:
            indices = torch.arange(self.batch_size)

        while True:
            x = torch.stack([torch.from_numpy(self.tokens[i:i+self.context_length].astype(np.int64)) for i in indices])
            y = torch.stack([torch.from_numpy(self.tokens[i+1:i+self.context_length+1].astype(np.int64)) for i in indices])
            yield x, y

            if self.shuffle:
                indices = torch.randint(0, len(self.tokens) - self.context_length, (self.batch_size,))
            else:
                indices += self.batch_size


def get_dataloader(file_path, tokenizer, batch_size, context_length, shuffle=True):
    """创建并返回一个 DataLoader（用于验证集）"""
    dataset = TextDataset(file_path, tokenizer, context_length, batch_size, shuffle)
    return DataLoader(dataset, batch_size=None, shuffle=False, drop_last=False)


# --- 学习率调度器 ---
def get_lr(it):
    """根据当前步数计算学习率（余弦退火 + 预热）。"""
    if it < training_config.warmup_steps:
        return training_config.learning_rate * it / training_config.warmup_steps
    if it > training_config.max_steps:
        return training_config.min_lr
    decay_ratio = (it - training_config.warmup_steps) / (training_config.max_steps - training_config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return training_config.min_lr + coeff * (training_config.learning_rate - training_config.min_lr)

# --- 评估函数 ---
@torch.no_grad()
def evaluate(model, dataloader, device):
    """在给定的 dataloader 上评估模型，返回平均损失。"""
    print("  -- 开始评估 --")
    model.eval()
    total_loss = 0
    eval_iters = training_config.eval_steps
    for i, (x, y) in enumerate(dataloader):
        if i >= eval_iters:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"    评估进度: {i+1}/{eval_iters}")

    model.train()
    print("  -- 评估完成 --")
    return total_loss / (eval_iters + 1e-6)

# --- 日志记录 ---
@contextmanager
def redirect_stdout_to_file(filepath):
    """同时输出到文件和终端的上下文管理器"""
    import sys

    class TeeOutput:
        def __init__(self, file, terminal):
            self.file = file
            self.terminal = terminal

        def write(self, message):
            self.file.write(message)
            self.file.flush()
            self.terminal.write(message)
            self.terminal.flush()

        def flush(self):
            self.file.flush()
            self.terminal.flush()

    original_stdout = sys.stdout
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            sys.stdout = TeeOutput(f, original_stdout)
            yield
    finally:
        sys.stdout = original_stdout

# ==============================================================================
# 训练接口
# ==============================================================================

def train_model(dataset_path: str, vocab_path: str, merge_path: str, ckpt_dir: str, chunk_size_mb: int = 10):
    """
    功能：
        - 使用流式编码从指定数据集训练 Transformer 模型
        - 边训练边编码，CPU和GPU并行工作
        - 自动保存 checkpoint (ckpt)，每2000步保存一次
        - 可选择从 ckpt 恢复继续训练

    参数：
        chunk_size_mb: 每个chunk的大小（MB），建议100-500MB
    """
    # --- 实验目录与日志设置 ---
    exp_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(ckpt_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_ctx = redirect_stdout_to_file(os.path.join(exp_dir, "log.txt")) if training_config.log_to_file else nullcontext()

    streaming_tokenizer = None  # 用于最后清理

    try:
        with log_ctx:
            print("=" * 80)
            print(f"开始新的训练实验 (流式编码模式)")
            print(f"实验目录: {exp_dir}")
            print(f"日志文件: {os.path.join(exp_dir, 'log.txt')}")
            print(f"Chunk大小: {chunk_size_mb} MB")
            print(f"Checkpoint保存间隔: 每2000步")
            print("=" * 80)

            # --- 设置 ---
            if training_config.device == 'cuda' and torch.cuda.is_available():
                device = f"cuda:{training_config.gpu_id}"
            else:
                device = "cpu"
            print(f"正在使用设备: {device}")

            torch.manual_seed(1337)

            # --- 数据 ---
            tokenizer = BPE()
            tokenizer.load_files(vocab_path, merge_path)
            print(f"已从 {vocab_path} 加载 BPE 模型。")

            # 使用流式加载训练集
            print("\n" + "=" * 80)
            print("初始化训练数据集（流式编码）")
            print("=" * 80)
            train_loader, streaming_tokenizer = get_streaming_dataloader(
                dataset_path, tokenizer, training_config.batch_size,
                model_config.context_length, chunk_size_mb=chunk_size_mb
            )

            # --- 模型与优化器 ---
            model_config.vocab_size = tokenizer.get_vocab_size()
            print(f"\n 模型词汇表大小已设置为: {model_config.vocab_size}")
            model = TransformerLM(**model_config.__dict__, device=device).to(device)

            total_params = sum(p.numel() for p in model.parameters())
            print(f"模型总参数量: {total_params:,} ({total_params/1e6:.2f}M)")

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay
            )

            # --- 从 Checkpoint 恢复 ---
            start_step = 0
            latest_ckpt_path = os.path.join(ckpt_dir, "latest.pt")
            if os.path.exists(latest_ckpt_path):
                print(f"\n🔄 正在从 checkpoint 恢复: {latest_ckpt_path}")
                checkpoint = torch.load(latest_ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_step = checkpoint['step'] + 1
                print(f"✅ 已恢复到步数 {start_step}")

            # --- 训练循环 ---
            print("\n" + "=" * 80)
            print(f"开始训练: 步数 {start_step} → {training_config.max_steps}")
            print(f"批量大小: {training_config.batch_size} | 上下文长度: {model_config.context_length}")
            print(f" 每步处理: {training_config.batch_size * model_config.context_length:,} tokens")
            print(f"后台CPU正在异步编码下一批数据...")
            print("=" * 80 + "\n")

            train_loader_iter = iter(train_loader)
            start_time = time.time()
            running_loss = 0.0
            log_loss_interval = 1000

            for step in range(start_step, training_config.max_steps):
                lr = get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                try:
                    x, y = next(train_loader_iter)
                except StopIteration:
                    train_loader_iter = iter(train_loader)
                    x, y = next(train_loader_iter)

                x, y = x.to(device), y.to(device)

                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if (step + 1) % log_loss_interval == 0:
                    avg_loss = running_loss / log_loss_interval
                    ppl = math.exp(avg_loss)
                    elapsed_time = time.time() - start_time
                    tokens_per_sec = (log_loss_interval * training_config.batch_size * model_config.context_length) / elapsed_time

                    print(f"步数 {step+1:6d}/{training_config.max_steps} | "
                          f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                          f"LR: {lr:.2e} | {tokens_per_sec:,.0f} tokens/s")

                    running_loss = 0.0
                    start_time = time.time()

                # 每2000步保存checkpoint
                if (step + 1) % training_config.save_interval == 0 or step + 1 == training_config.max_steps:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': step,
                        'model_config': model_config.__dict__
                    }
                    ckpt_path = os.path.join(exp_dir, f"step_{step}.pt")
                    torch.save(checkpoint, ckpt_path)
                    print(f"模型已保存到: {ckpt_path}")
                    torch.save(checkpoint, latest_ckpt_path)

            print("\n" + "=" * 80)
            print("训练完成！")
            print(f"所有文件已保存到: {exp_dir}")
            print("=" * 80)

    finally:
        # 清理后台线程
        if streaming_tokenizer is not None:
            print("\n停止后台编码线程...")
            streaming_tokenizer.stop()
            print("后台线程已停止")


def infer(model_ckpt: str, vocab_path: str, merge_path: str, test_dataset: str, max_new_tokens: int = 50):
    """推理接口"""
    print(f"正在从 {model_ckpt} 加载模型...")
    if training_config.device == 'cuda' and torch.cuda.is_available():
        device = f"cuda:{training_config.gpu_id}"
    else:
        device = "cpu"
    print(f"正在使用设备: {device}")

    checkpoint = torch.load(model_ckpt, map_location=device)
    ckpt_model_config = argparse.Namespace(**checkpoint['model_config'])

    tokenizer = BPE()
    tokenizer.load_files(vocab_path, merge_path)
    print(f"已从 {vocab_path} 加载 BPE 模型。")

    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    ckpt_model_config.vocab_size = tokenizer.get_vocab_size()

    model = TransformerLM(**ckpt_model_config.__dict__, device=device).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("正在计算测试集上的 PPL...")
    test_loader = get_dataloader(test_dataset, tokenizer, training_config.batch_size, ckpt_model_config.context_length, shuffle=False)
    test_loss = evaluate(model, test_loader, device)
    ppl = math.exp(test_loss)
    print(f"在测试集 '{os.path.basename(test_dataset)}' 上的最终困惑度 (PPL): {ppl:.4f}")

    results = []
    num_samples_to_gen = 3
    print(f"\n--- 正在从测试集生成 {num_samples_to_gen} 个样本 ---")

    with torch.no_grad():
        test_loader_iter = iter(test_loader)
        for i in range(num_samples_to_gen):
            try:
                x, y = next(test_loader_iter)
            except StopIteration:
                break

            input_ids = x[0:1].to(device)
            prompt_tokens = [id_to_token.get(idx.item(), "<UNK>") for idx in input_ids[0]]
            prompt_text = tokenizer.decode(prompt_tokens)

            generated_ids = input_ids.tolist()[0]
            generated_probs = []

            for _ in range(max_new_tokens):
                input_crop = input_ids[:, -ckpt_model_config.context_length:]
                logits, _ = model(input_crop)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

                generated_ids.append(next_id.item())
                generated_probs.append(probs[0, next_id.item()].item())
                input_ids = torch.cat((input_ids, next_id), dim=1)

            generated_tokens = [id_to_token.get(gid, "<UNK>") for gid in generated_ids]

            results.append({
                "input_text": prompt_text,
                "tokens": generated_tokens,
                "probs": generated_probs
            })
            print(f"\n样本 {i+1} | 输入: '{prompt_text[:50]}...'")
            print(f"样本 {i+1} | 生成: '{tokenizer.decode(generated_tokens)}'")

    return results, ppl


def main():
    """主函数，用于解析命令行参数并调用相应的功能。"""
    parser = argparse.ArgumentParser(description="CS336 Assignment 1: Streaming Transformer Training")
    parser.add_argument("--train", action="store_true", help="启动训练模式")
    parser.add_argument("--infer", action="store_true", help="启动推理模式")

    parser.add_argument("--dataset", type=str, default=data_config.TRAIN_DATASET_PATH, help="训练数据集路径")
    parser.add_argument("--vocab", type=str, default=data_config.VOCAB_PATH, help="词汇表文件路径")
    parser.add_argument("--merge", type=str, default=data_config.MERGE_PATH, help="BPE 合并规则文件路径")
    parser.add_argument("--ckpt_dir", type=str, default=data_config.CKPT_DIR, help="Checkpoint 保存目录")
    parser.add_argument("--ckpt", type=str, default=os.path.join(data_config.CKPT_DIR, "latest.pt"), help="用于推理的模型 checkpoint 路径")
    parser.add_argument("--test", type=str, default=data_config.TEST_DATASET_PATH, help="测试数据集路径")
    parser.add_argument("--gpu_id", type=int, default=training_config.gpu_id, help="要使用的 GPU 编号")
    parser.add_argument("--max_new_tokens", type=int, default=inference_config.max_new_tokens, help="推理时生成的最大 token 数")
    parser.add_argument("--chunk_size_mb", type=int, default=100, help="流式编码的chunk大小（MB）")

    parser.add_argument("--device", choices=["cpu", "cuda"], default=training_config.device)
    parser.add_argument("--max_steps", type=int, default=training_config.max_steps)
    parser.add_argument("--batch_size", type=int, default=training_config.batch_size)
    args = parser.parse_args()

    training_config.gpu_id = args.gpu_id
    training_config.device = args.device
    training_config.max_steps = args.max_steps
    training_config.batch_size = args.batch_size
    training_config.warmup_steps = min(training_config.warmup_steps, max(1, args.max_steps // 10))
    if args.device == "cuda" and (not torch.cuda.is_available() or args.gpu_id >= torch.cuda.device_count()):
        parser.error("Requested CUDA device is unavailable; use --device cpu or a valid --gpu_id")
    if args.max_steps < 1 or args.batch_size < 1 or args.gpu_id < 0 or args.chunk_size_mb < 1:
        parser.error("steps, batch size and chunk size must be positive; GPU ID must be nonnegative")
    if args.train or args.infer:
        BPE().load_files(args.vocab, args.merge)
        dataset = args.dataset if args.train else args.test
        if not os.path.isfile(dataset) or os.path.getsize(dataset) == 0:
            parser.error(f"Dataset is missing or empty: {dataset}")

    if args.train:
        print("--- 启动训练模式（流式编码）---")
        train_model(
            dataset_path=args.dataset,
            vocab_path=args.vocab,
            merge_path=args.merge,
            ckpt_dir=args.ckpt_dir,
            chunk_size_mb=args.chunk_size_mb
        )
    elif args.infer:
        print("--- 启动推理模式 ---")
        infer(
            model_ckpt=args.ckpt,
            vocab_path=args.vocab,
            merge_path=args.merge,
            test_dataset=args.test,
            max_new_tokens=args.max_new_tokens
        )
    else:
        print("请指定模式: --train 或 --infer")

if __name__ == "__main__":
    main()
