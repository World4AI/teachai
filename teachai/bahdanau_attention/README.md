# Bahdanau Attention

This folder contains the PyTorch implementation for 'Bahdanau Attention', based on the paper [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473). The implementation is loosely based on this PyTorch [tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html).
In this example we use the German - English bilingual sentence pairs dataset from [manythings.org](http://www.manythings.org/anki/).

You can use the following `bash` commands to dowload and unzip the data, if you want to train the model from scratch.

```bash
mkdir downloads
cd downloads/
curl -O https://www.manythings.org/anki/deu-eng.zip
unzip deu-eng.zip
cd -
```

Our implementation does not fully reflect the model from the paper, but we try to keep the general idea. We provide pretrained weights for a model that we trained for 40 epochs. After 40 epochs we can achieve the following results.

```bash
--------------------------------------------------
Generating translation for: I am hungry
Model Translation: "ich habe hunger"
Expected Translation: Ich bin hungrig
--------------------------------------------------
Generating translation for: She is funny
Model Translation: "sie ist faul geworden"
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
Model Translation: "ich bin wandern gegangen"
Expected Translation: Ich ging zur Uni
--------------------------------------------------
Generating translation for: We have many options
Model Translation: "wir haben viele möglichkeiten <unk>"
Expected Translation: Wir haben viele Möglichkeiten
--------------------------------------------------
Generating translation for: Life is not fair
Model Translation: "das leben ist nicht gerecht"
Expected Translation: Das Leben ist ungerecht
--------------------------------------------------
Generating translation for: To be or not to be
Model Translation: "sei oder nicht wahr"
Expected Translation: Sein oder nicht sein
```

While some of the translations are quite accurate, others are lacking. For example the sentence _She is funny_ becomes _She became lazy._ This is expected, as the dataset and the model are relatively small for the task.
