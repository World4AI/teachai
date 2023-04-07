# Transformer

This folder contains the PyTorch implementation of the original transformer network, based on the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

The implementation is inspired by the following tutorials.

1. [LANGUAGE TRANSLATION WITH NN.TRANSFORMER AND TORCHTEXT](https://pytorch.org/tutorials/beginner/translation_transformer.html)
2. [Notebook by Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)
3. [Transformer Anatomy](https://github.com/nlp-with-transformers/notebooks/blob/main/03_transformer-anatomy.ipynb)

In this example we use the German - English bilingual sentence pairs dataset from [manythings.org](http://www.manythings.org/anki/). You can use the following `bash` commands to dowload and unzip the data, if you want to train the model from scratch.

```bash
mkdir downloads
cd downloads/
curl -O https://www.manythings.org/anki/deu-eng.zip
unzip deu-eng.zip
cd -
```

We trained the model for 10 epochs and get the following example translations. The results are not perfect, but given more time and data, we could achieve much better results if we scaled up the model.

```
--------------------------------------------------
Generating translation for: I am hungry
Model Translation: "ich habe hunger"
Expected Translation: Ich bin hungrig
--------------------------------------------------
Generating translation for: She is funny
Model Translation: "sie ist lustig"
Expected Translation: Sie ist witzig
--------------------------------------------------
Generating translation for: I need a new car!
Model Translation: "ich brauche ein neues auto"
Expected Translation: Ich brauche ein neues Auto
--------------------------------------------------
Generating translation for: It is extremely hot
Model Translation: "es ist äußerst heiß"
Expected Translation: It is extrem heiß
--------------------------------------------------
Generating translation for: I went to college
Model Translation: "ich bin <unk>"
Expected Translation: Ich ging zur Uni
--------------------------------------------------
Generating translation for: We have many options
Model Translation: "wir haben viele möglichkeiten"
Expected Translation: Wir haben viele Möglichkeiten
--------------------------------------------------
Generating translation for: Life is not fair
Model Translation: "das leben ist nicht fair"
Expected Translation: Das Leben ist ungerecht
--------------------------------------------------
Generating translation for: To be or not to be
Model Translation: "es wird oder nicht zu sein"
Expected Translation: Sein oder nicht sein

```
