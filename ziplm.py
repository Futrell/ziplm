import gzip
import bz2
import lzma

import numpy as np
import scipy.special

class ZipModel:
    def __init__(self, vocabulary, training="", compressor=gzip):
        self.vocabulary = vocabulary
        self.training = training
        self.compressor = compressor
        self.index = {v:i for i, v in enumerate(self.vocabulary)}

    def logprobs(self, prefix="", temperature=1):
        code_lengths = np.array([
            len(self.compressor.compress("".join([self.training, prefix, v]).encode()))
            for v in self.vocabulary
        ])
        return scipy.special.log_softmax(-code_lengths*np.log(len(self.vocabulary))*(1/temperature))
                                         
    def sequence_logprob(self, sequence, prefix="", temperature=1):
        score = 0.0
        for x in sequence:
            scores = self.logprobs(prefix, temperature=temperature)
            score += scores[self.index[x]]
            prefix += x
        return score

    def sample(self, prefix="", temperature=1):
        scores = self.logprobs(prefix, temperature=temperature)
        i = np.random.choice(range(len(self.vocabulary)), p=np.exp(scores))
        return self.vocabulary[i]

    def sample_sequence(self, maxlen, prefix="", temperature=1):
        sequence = prefix
        for k in range(maxlen):
            result = self.sample(sequence, temperature=temperature)
            yield result
            sequence += result

