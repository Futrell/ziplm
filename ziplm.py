import gzip
import bz2
import lzma

import numpy as np
import scipy.special

class ZipModel:
    def __init__(self, vocabulary, training="", conversion=np.log(2), compressor=gzip):
        self.vocabulary = vocabulary
        self.training = training
        self.compressor = compressor
        self.index = {v:i for i, v in enumerate(self.vocabulary)}

    def logprobs(self, prefix=""):
        code_lengths = np.array([
            len(self.compressor.compress("".join([self.training, prefix, v]).encode()))
            for v in self.vocabulary
        ])
        return scipy.special.log_softmax(-code_lengths*np.log(len(self.vocabulary)), -1)

    def sequence_logprob(self, sequence):
        prefix = []
        score = 0.0
        for x in sequence:
            scores = self.logprobs("".join(prefix))
            score += scores[self.index[x]]
            prefix.append(x)
        return score

    def sample(self, prefix=""):
        scores = self.logprobs(prefix)
        i = np.random.choice(range(len(self.vocabulary)), p=np.exp(scores))
        return self.vocabulary[i]

    def sample_sequence(self, prefix, maxlen=None):
        sequence = [prefix]
        for k in range(maxlen):
            result = self.sample("".join(sequence))
            yield result
            sequence.append(result)

