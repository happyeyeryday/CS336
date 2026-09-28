"""
BPE (Byte Pair Encoding) Implementation
提交位置: /data/nlp_course/assignments/hw1/{user}/
"""
import pickle
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
import re # 导入 re 模块


class BPE:
    def __init__(self):
        """
        初始化 BPE 模型。
        - self.vocab: 词汇表 dict[str, int] 记录每个token的ID
        - self.merges: 合并规则 list[tuple[str, str]] 按优先级排序
        - self.bpe_codes: 快速查找的合并映射 dict[tuple[str, str], str]
        """
        self.vocab = {}
        self.merges = []
        self.bpe_codes = {}

    def _get_vocab(self, corpus: List[str]) -> Dict[Tuple[str, ...], int]:
        """
        构建初始词汇表，每个词被拆分为字符序列，末尾加上</w>标记

        参数：
            corpus: 输入语料列表
        返回：
            vocab: dict，key为字符元组，value为词频
        """
        vocab = defaultdict(int)
        for word in corpus:
            # 将每个词拆分为字符，末尾添加</w>
            word_tuple = tuple(list(word) + ['</w>'])
            vocab[word_tuple] += 1
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
        # 获取初始词汇表
        vocab = self._get_vocab(corpus)

        # 执行 num_merges 次合并
        for i in range(num_merges):
            # 统计符号对频率
            pairs = self._get_stats(vocab)

            if not pairs:
                break

            # 找出频率最高的符号对
            best_pair = max(pairs, key=pairs.get)

            # 记录合并规则
            self.merges.append(best_pair)

            # 执行合并
            vocab = self._merge_vocab(best_pair, vocab)

        # 构建 bpe_codes 用于快速查找
        self.bpe_codes = {pair: ''.join(pair) for pair in self.merges}

        # 构建最终词汇表（所有可能的token）
        self.vocab = {}
        idx = 0

        # 添加所有基础字符
        base_chars = set()
        for word_tuple in vocab.keys():
            base_chars.update(word_tuple)

        for char in sorted(base_chars):
            self.vocab[char] = idx
            idx += 1

    def _tokenize_word(self, word: str) -> List[str]:
        """
        对单个词进行BPE分词

        参数：
            word: 输入词
        返回：
            tokens: BPE子词列表
        """
        # 初始化为字符序列 + </w>
        tokens = list(word) + ['</w>']

        # --- 优化点：避免在循环中修改列表 ---
        # 循环直到没有可合并的 pair
        while True:
            # 统计当前 tokens 中的 pairs
            pairs = Counter()
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i+1])] += 1

            # 找到在 self.merges 中优先级最高的、且存在于当前 tokens 中的 pair
            best_pair = None
            min_rank = float('inf')
            for pair, rank in self.merges_rank.items():
                if rank < min_rank and pairs[pair] > 0:
                    min_rank = rank
                    best_pair = pair

            # 如果找不到可合并的 pair，则结束循环
            if best_pair is None:
                break

            # 执行合并
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(''.join(best_pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        对输入文本进行分词，返回 token ID 序列。

        参数：
            text: str, 输入文本
        返回：
            token_ids: list[int], BPE token ID 序列
        """
        if not self.merges:
            raise ValueError("模型未训练，请先调用 train() 方法")

        token_ids = []
        # --- 优化点：使用正则表达式迭代器，避免 text.split() 内存爆炸 ---
        # 这个正则表达式会匹配一个或多个非空格字符（单词）
        for match in re.finditer(r'\S+', text):
            word = match.group(0)
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                if token in self.vocab:
                    token_ids.append(self.vocab[token])
                # 对于未登录词，可以忽略或映射到一个未知 token

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
        with open(f"{path_prefix}_merges.pkl", "rb") as f:
            self.merges = pickle.load(f)
        with open(f"{path_prefix}_vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)

        # 重建 bpe_codes 和 merges_rank
        self.bpe_codes = {pair: ''.join(pair) for pair in self.merges}
        self.merges_rank = {pair: i for i, pair in enumerate(self.merges)}

    def get_vocab_size(self) -> int:
        """返回词汇表大小"""
        return len(self.vocab)

    def get_num_merges(self) -> int:
        """返回合并规则数量"""
        return len(self.merges)


def main():
    bpe = BPE()
    corpus = ["low", "lower", "lowest", "the", "quick", "brown", "fox"]
    bpe.train(corpus, num_merges=10)
