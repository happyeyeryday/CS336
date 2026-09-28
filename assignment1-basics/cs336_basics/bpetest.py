from .bpe import BPE

bpe = BPE()
corpus = ["low", "lower", "lowest", "the", "quick", "brown", "fox"]
bpe.train(corpus, num_merges=10)
