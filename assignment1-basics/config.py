# config.py
from dataclasses import dataclass
import os

# -----------------------------------------------------------------------------
# 配置说明:
# 老师已指定训练集为 owt_train.txt (~12GB)。
# 以下配置根据老师给出的 50M 模型性能基线 (90-100k tokens/s) 进行了调整。
# -----------------------------------------------------------------------------

@dataclass
class DataConfig:
    """数据与路径相关配置"""
    # ------------------- BPE Tokenizer 相关路径 -------------------
    VOCAB_PATH: str = "./data/bpe_model_full_vocab.pkl"
    MERGE_PATH: str = "./data/bpe_model_full_merges.pkl"

    # ------------------- 数据集相关路径 (已根据老师要求更新) -------------------
    # 训练数据集的 .txt 文件路径
    TRAIN_DATASET_PATH: str = "./data/owt_train.txt"
    # 验证数据集路径，用于在训练中监控PPL
    VALIDATION_DATASET_PATH: str = "./data/owt_valid.txt"
    # 测试数据集路径 (老师未指定，通常用验证集代替)
    TEST_DATASET_PATH: str = "./data/owt_valid.txt"

    # ------------------- 输出路径 -------------------
    # 存放模型 checkpoints 的目录
    CKPT_DIR: str = "checkpoints"

# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """
    模型架构相关配置。
    当前的配置 (d_model=512, n_layers=8) 大约是一个 50M 参数的模型，与老师的基线匹配，是一个很好的起点。
    """
    # 老师参考了 GPT-2 的 50k 词表，所以我们将默认值设为 GPT-2 的标准词汇表大小。
    # 最终值仍取决于你队友在 owt_train.txt 上训练BPE的结果。
    vocab_size: int = 40375

    context_length: int = 384
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0

# -----------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """
    训练过程相关配置。
    这是根据老师给出的性能基线估算得到的，是调优的核心。
    """
    # 'cuda' 表示使用 GPU，'cpu' 表示使用 CPU。
    # 最终使用的设备将是 f"cuda:{gpu_id}"
    device: str = "cuda"
    gpu_id: int = 0  # 指定要使用的 GPU 编号

    # ------------------- 优化器参数 (AdamW) -------------------
    # 将 batch_size 从 128 减小到 64，以降低显存占用
    batch_size: int = 96           # 批量大小。在A100上可以适当调大以提高吞吐量。
    learning_rate: float = 6e-4   # 最大学习率，非常敏感，需要仔细调整。
    weight_decay: float = 0.1     # 权重衰减

    # ------------------- 训练循环与时间管理 -------------------
    # !!! 关键参数: 由于 batch_size 减半，为了在相同时间内处理相似数量的 token，需要将 max_steps 加倍 !!!
    # 计算过程:
    #   - 10 小时 = 36000 秒
    #   - 速度 ≈ 95,000 tokens/秒
    #   - 每个训练步的 tokens 数 = batch_size * context_length = 64 * 256 = 16384
    #   - 总步数 ≈ (36000 * 95000) / 16384 ≈ 208,720 步
    # 我们设置一个整数。
    max_steps: int = 208000

    # ------------------- 学习率调度器 (Cosine Annealing) -------------------
    warmup_steps: int = 2000
    min_lr: float = 6e-5

    # ------------------- 日志与 Checkpointing -------------------
    log_to_file: bool = True     # 是否将日志输出到文件
    log_interval: int = 50       # 增大了打印间隔，避免日志刷屏
    eval_interval: int = 1000
    eval_steps: int = 200        # 评估时迭代的步数，避免评估时间过长
    save_interval: int = 1000

# -----------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """推理过程相关配置"""
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_p: float = 0.9

# -----------------------------------------------------------------------------
# 实例化配置类
# -----------------------------------------------------------------------------
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
inference_config = InferenceConfig()
