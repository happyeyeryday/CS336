"""
BPE (Byte Pair Encoding) Implementation - 优化版
"""
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
import re
import heapq  # 导入 heapq 用于优先队列
from tqdm import tqdm


class BPE:
    def __init__(self):
        """
        初始化 BPE 模型。
        - self.vocab: 词汇表 dict[str, int] 记录每个token的ID
        - self.merges: 合并规则 list[tuple[str, str]] 按优先级排序
        - self.bpe_codes: 快速查找的合并映射 dict[tuple[str, str], str]
        - self.merges_rank: 合并规则的优先级 dict[tuple[str, str], int]
        """
        self.vocab = {}
        self.merges = []
        self.bpe_codes = {}
        self.merges_rank = {}

    def _get_vocab(self, corpus: List[str]) -> Dict[Tuple[str, ...], int]:
        """
        构建初始词汇表，每个词被拆分为字符序列，末尾加上</w>标记

        参数：
            corpus: 输入语料列表
        返回：
            vocab: dict，key为字符元组，value为词频
        """
        vocab = defaultdict(int)

        print(f"📊 语料规模: {len(corpus):,} 个词")

        # 批处理
        batch_size = 100000

        with tqdm(total=len(corpus), desc="📝 构建词汇", unit="词") as pbar:
            for i in range(0, len(corpus), batch_size):
                batch = corpus[i:i + batch_size]

                for word in batch:
                    # 将每个词拆分为字符，末尾添加</w>
                    word_tuple = tuple(list(word) + ['</w>'])
                    vocab[word_tuple] += 1

                pbar.update(len(batch))

        print(f" 去重后: {len(vocab):,} 个词型 (压缩 {len(corpus)/len(vocab):.1f}x)")
        return vocab

    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Counter:
        """
        统计所有相邻符号对的出现频率

        参数：
            vocab: 当前词汇表
        返回：
            pairs: Counter对象，统计每个符号对的频率
        """
        pairs = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """
        将词汇表中所有出现的符号对进行合并

        参数：
            pair: 要合并的符号对
            vocab: 当前词汇表
        返回：
            new_vocab: 合并后的新词汇表
        """
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in vocab.items():
            # 将元组转换为字符串进行替换
            word_str = ' '.join(word)
            new_word_str = word_str.replace(bigram, replacement)
            # 转换回元组
            new_word = tuple(new_word_str.split())
            new_vocab[new_word] = freq

        return new_vocab

    def train(self, corpus: List[str], num_merges: int) -> None:
        """
        训练 BPE 模型：
        1. 将语料拆分为初始符号序列（字符级别 + </w>）
        2. 统计符号对的出现频率
        3. 找出最频繁的一对并合并
        4. 重复 num_merges 次
        5. 保存最终的 merges 和 vocab

        参数：
            corpus: list[str]，训练语料，每个元素是一个词
            num_merges: int，执行的合并次数
        """
        import time

        if not corpus or num_merges < 1:
            raise ValueError("corpus must be nonempty and num_merges must be positive")
        self.merges = []
        # Keep the initial alphabet even when every occurrence was merged away.
        initial_tokens = set(''.join(corpus)) | {'</w>'}

        # 获取初始词汇表
        print("\n" + "="*60)
        print("📊 构建初始词汇表...")
        print("="*60)
        start_time = time.time()
        vocab = self._get_vocab(corpus)
        init_time = time.time() - start_time
        init_vocab_size = len(vocab)

        print(f"\n 初始词汇构建完成")
        print(f"   - 不重复词型数: {init_vocab_size:,}")
        print(f"   - 耗时: {init_time:.2f}秒")

        ratio = len(corpus) / init_vocab_size
        estimated_per_merge = init_time / ratio
        total_estimated = estimated_per_merge * num_merges

        print(f"\n  训练时间预估:")
        print(f"   - 去重比例: {ratio:.1f}x")
        print(f"   - 预计每次merge: {estimated_per_merge:.2f}秒")
        if total_estimated < 3600:
            print(f"   - 预计总时间: {total_estimated/60:.1f}分钟")
        else:
            print(f"   - 预计总时间: {total_estimated/3600:.1f}小时")

        print("="*60)
        print()

        # 执行 num_merges 次合并
        actual_times = []
        with tqdm(total=num_merges, desc="🔥 训练BPE", unit="merge") as pbar:
            for i in range(num_merges):
                iter_start = time.time()

                # 统计符号对频率
                pairs = self._get_stats(vocab)

                if not pairs:
                    pbar.write(f"⚠️  第 {i} 轮：没有更多可合并的符号对，提前结束")
                    break

                # 找出频率最高的符号对
                best_pair = max(pairs, key=pairs.get)
                best_freq = pairs[best_pair]

                # 记录合并规则
                self.merges.append(best_pair)

                # 执行合并
                vocab = self._merge_vocab(best_pair, vocab)

                iter_time = time.time() - iter_start
                actual_times.append(iter_time)

                # 每100轮输出一次详细信息
                if (i + 1) % 100 == 0:
                    compression_ratio = init_vocab_size / len(vocab)
                    avg_time = sum(actual_times[-100:]) / len(actual_times[-100:])
                    remaining = num_merges - (i + 1)
                    eta_seconds = remaining * avg_time

                    if eta_seconds < 3600:
                        eta_str = f"{eta_seconds/60:.1f}分钟"
                    else:
                        eta_str = f"{eta_seconds/3600:.1f}小时"

                    pbar.write(f"📈 第 {i+1} 轮: 压缩率 {compression_ratio:.2f}x, "
                              f"实际速度 {avg_time:.2f}s/merge, 预计剩余 {eta_str}")

                # 更新进度条
                pair_display = f"{best_pair[0]}+{best_pair[1]}"
                if len(pair_display) > 15:
                    pair_display = pair_display[:15] + "..."

                pbar.set_postfix({
                    'pair': pair_display,
                    'freq': f"{best_freq:,}",
                    'vocab': len(vocab)
                })
                pbar.update(1)

        # 构建 bpe_codes 用于快速查找
        self.bpe_codes = {pair: ''.join(pair) for pair in self.merges}

        # 🚀 构建 merges_rank 用于优先级查找
        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}

        # 构建最终词汇表（所有可能的token）
        print("\n🔨 构建最终词汇表...")
        self.vocab = {}
        idx = 0

        # 收集所有基础字符（初始字符级 token）
        base_tokens = initial_tokens.copy()
        for word_tuple in vocab.keys():
            base_tokens.update(word_tuple)

        # 收集所有合并产生的新 token
        merged_tokens = set()
        for a, b in self.merges:
            merged_tokens.add(a + b)

        # 汇总所有可能的 token
        all_tokens = sorted(base_tokens.union(merged_tokens))

        # 编号
        for token in all_tokens:
            self.vocab[token] = idx
            idx += 1

        print(f"\n✅ 训练完成！")
        print(f"   - 合并规则: {len(self.merges):,} 条")
        print(f"   - 最终词汇: {len(self.vocab):,} 个token")
        print(f"   - 总耗时: {time.time() - start_time:.2f}秒")
        print("="*60 + "\n")

    def _tokenize_word(self, word: str) -> List[str]:
        """
        对单个词进行BPE分词 (使用优先队列优化)

        参数：
            word: 输入词
        返回：
            tokens: BPE子词列表

        时间复杂度: O(n log n) 其中 n = 词长度
        """
        if not word:
            return []

        # 初始化为字符序列 + </w>
        tokens = list(word) + ['</w>']

        if len(tokens) == 1:
            return tokens

        # 🚀 使用优先队列存储所有可合并的pair
        # 格式: (优先级rank, 位置pos, pair)
        heap = []

        # 初始化: 将所有相邻pair加入堆
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            if pair in self.merges_rank:
                rank = self.merges_rank[pair]
                heapq.heappush(heap, (rank, i, pair))

        # 标记哪些位置已被合并
        merged = [False] * len(tokens)

        while heap:
            rank, pos, pair = heapq.heappop(heap)

            # 跳过已经失效的pair (位置已被合并)
            if merged[pos] or merged[pos + 1]:
                continue

            # 验证pair仍然有效
            if pos < len(tokens) - 1 and tokens[pos] == pair[0] and tokens[pos+1] == pair[1]:
                # 执行合并
                tokens[pos] = pair[0] + pair[1]
                merged[pos + 1] = True

                # 检查新产生的相邻pair - 左侧
                if pos > 0 and not merged[pos - 1]:
                    new_pair = (tokens[pos - 1], tokens[pos])
                    if new_pair in self.merges_rank:
                        new_rank = self.merges_rank[new_pair]
                        heapq.heappush(heap, (new_rank, pos - 1, new_pair))

                # 检查新产生的相邻pair - 右侧
                if pos + 2 < len(tokens) and not merged[pos + 2]:
                    new_pair = (tokens[pos], tokens[pos + 2])
                    if new_pair in self.merges_rank:
                        new_rank = self.merges_rank[new_pair]
                        heapq.heappush(heap, (new_rank, pos, new_pair))

        # 收集未被合并的token
        result = [tokens[i] for i in range(len(tokens)) if not merged[i]]
        return result

    def encode(self, text: str) -> List[str]:
        """

        对输入文本进行分词，返回 token 字符串序列。

        参数：
            text: str, 输入文本
        返回：
            tokens: list[str], BPE token 字符串序列
        """
        if not self.merges:
            raise ValueError("模型未训练，请先调用 train() 或 load() 方法")

        tokens = []
        # 使用正则表达式迭代器，避免 text.split() 内存爆炸
        for match in re.finditer(r'\S+', text):
            word = match.group(0)
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)  # 直接添加字符串token

        return tokens

    def encode_to_ids(self, text: str) -> List[int]:
        """
        对输入文本进行分词，返回 token ID 序列。

        参数：
            text: str, 输入文本
        返回：
            token_ids: list[int], BPE token ID 序列
        """
        tokens = self.encode(text)
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            # 对于未登录词，可以忽略或映射到一个未知 token (这里选择忽略)
        return token_ids

    def decode(self, tokens: List[str]) -> str:
        """
        将子词序列还原为原始字符串。

        参数：
            tokens: list[str]
        返回：
            text: str
        """
        # 合并所有token，移除</w>标记并替换为空格
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        return text.strip()

    def save(self, path_prefix: str) -> None:
        """
        保存 merges 与 vocab。
        输出：
            {path_prefix}_merges.pkl
            {path_prefix}_vocab.pkl
        """
        Path(path_prefix).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{path_prefix}_merges.pkl", "wb") as f:
            pickle.dump(self.merges, f)
        with open(f"{path_prefix}_vocab.pkl", "wb") as f:
            pickle.dump(self.vocab, f)

    def load(self, path_prefix: str) -> None:
        """
        从 pkl 文件中加载 merges 与 vocab。
        输入：
            path_prefix: 文件名前缀（不含扩展名）
        """
        self.load_files(f"{path_prefix}_vocab.pkl", f"{path_prefix}_merges.pkl")

    def load_files(self, vocab_path: str, merge_path: str) -> None:
        """Load an explicit pair of locally generated tokenizer files."""
        for path in (vocab_path, merge_path):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"Tokenizer file not found: {path}. Run examples/train_tokenizer.py first."
                )
        with open(merge_path, "rb") as f:
            self.merges = pickle.load(f)
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        # 重建 bpe_codes 和 merges_rank
        self.bpe_codes = {pair: ''.join(pair) for pair in self.merges}
        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}

        print(f"模型已加载:")
        print(f"  - 合并规则: {len(self.merges)} 条")
        print(f"  - 词汇表大小: {len(self.vocab)}")

    def get_vocab_size(self) -> int:
        """返回词汇表大小"""
        return len(self.vocab)

    def get_num_merges(self) -> int:
        """返回合并规则数量"""
        return len(self.merges)
