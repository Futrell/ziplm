# ziplm

Useless but mildly interesting language model using compressors built-in to Python.

## Usage

You can "train" it using some training data:

```{python}
data = open(my_favorite_text_file).read().lower()
alphabet = "qwertyuiopasdfghjklzxcvbnm,.;1234567890 "
model = ziplm.ZipModel(alphabet, training=data)
"".join(model.sample_sequence(10)) # sample 10 characters from the alphabet
```

You can also run it without any training data, and just forward sample to see what kinds of patterns gzip likes:
```{python}
alphabet = "qwertyuiopasdfghjklzxcvbnm "
model = ziplm.ZipModel(alphabet)
"".join(model.sample_sequence(100)) # I get 'oodegeeup ewjlrm chuzjqvrm chukkzfcjmwndxnlrm o qewmxweqswdqrlfiuyxlygxkxinsvrwrfjr ywmzwc mxhqerxzy'
```

You can also get the probability for a sequence:
```{python}
alphabet = "qwertyuiopasdfghjklzxcvbnm "
model = ziplm.ZipModel(alphabet)
model.sequence_logprob("this is my favorite string") # I get -83.8
```

You can also try using `bz2` and `lzma` as language models by passing them as the `compressor` argument to the model

```{python}
import lzma
model = ziplm.ZipModel(alphabet, compressor=lzma)
"".join(model.sample_sequence(100)) # I get 'pvmucalppovpriitgztwbwgksgphhnhdf huykoppunlrh izjbokivpcx eagjqiyfmaibjconivxobcmipdccwqoyiwxzisgzx'
```

## Why does this work?

This works because of two facts:
1. A language model is nothing but a distribution on the next token given previous tokens, $p(x \mid c)$.
2. There is a general equivalence between *probability distributions* and *codes*.

The second point is what makes this interesting. Information theory tells us that we can derive codes from probability distributions. That is, if I have some datapoints $x$, and I know that they follow probability distribution $p(x)$, I can come up with a lossless binary code to encode the $x$ where the length of each code is $-\log_2 p(x)$. This code minimizes the average code length: the only way to get shorter average code length would be to go into the realm of lossy compression. This is called the Shannon Limit.

Since I can convert probability distributions to codes in this way, I can also convert codes to probability distributions. If I have a code (like gzip) that describes my datapoint with length $l(x)$ in binary, then that corresponds to a probability distribution $p(x) = 2^{-l(x)}$. If the code is $K$-ary, then the corresponding distribution is 
$$p(x) = K^{-l(x)}.$$ 

The ZipLM model works by converting code lengths to probabilities in this way. If I have a vocabulary of size $K$, and a string $c$, then the probability distribution for continuations $x$ is:
$$p(x \mid c) \propto K^{-l(cx)},$$
where the proportionality reflects the fact that we have to sum over the compressed lengths of $cx^\prime$ for all $x^\prime$ in the vocabulary. That's all there is to it.



