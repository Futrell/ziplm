# ziplm

Useless but mildly interesting language model using compressors built-in to Python.

## Usage

You can "train" it using some training data:

```{python}
data = open(my_favorite_text_file).read().lower()
alphabet = "qwertyuiopasdfghjklzxcvbnm,.;1234567890 "
model = ziplm.ZipModel(alphabet, training=data)
model.sample_sequence(10) # sample 10 characters from the alphabet
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

