import time
import pathlib
import re
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import Dataset, DataLoader

# parameters
input_folder = "downloads"
input_file = "deu.txt"
seed = 42  # make training more deterministic
max_len = 10  # maximum allowed length of a sentence
min_freq = 5  # the minimum amount of times a word has to occur in order to be included
batch_size = 128  # batch size for data loader
enc_hidden_size = 128  # encoder lstm hidden layer size
dec_hidden_size = enc_hidden_size * 2  # decoder lstm hidden layer size
enc_nlayers = 2  # number of layers in the encoder lstm
dec_nlayers = 2  # number of layers in the decoder lstm
num_epochs = 40
lr = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_training = False  # train or evaluate


if __name__ == "__main__":
    torch.manual_seed(seed)

    # Loading and Cleaning Dataset
    def tokenize(s):
        s = s.lower().strip()  # lowercase and remove white space
        s = re.sub(r'[.,"\'-?:!;]', "", s)  # remove punctuation
        s = s.split(" ")  # split string into list of words
        s.insert(0, "<sos>")  # prepend start of sentence token
        s.append("<eos>")  # append end of sequence token
        return s

    path = pathlib.Path(input_folder) / input_file
    lang1, lang2 = [], []
    with open(path, "r") as f:
        for line in f:
            # the third element is the licence
            sen_1, sen_2, _ = line.split("\t")
            sen_1, sen_2 = tokenize(sen_1), tokenize(sen_2)

            if len(sen_1) <= max_len and len(sen_2) <= max_len:
                lang1.append(sen_1)
                lang2.append(sen_2)

    # create vocabulary
    lang1_counter = Counter()
    lang2_counter = Counter()

    for sen_1, sen_2 in zip(lang1, lang2):
        lang1_counter.update(sen_1)
        lang2_counter.update(sen_2)

    lang1_sorted = sorted(lang1_counter.items(), key=lambda x: x[1], reverse=True)
    lang2_sorted = sorted(lang2_counter.items(), key=lambda x: x[1], reverse=True)
    lang1_ordered = OrderedDict(lang1_sorted)
    lang2_ordered = OrderedDict(lang2_sorted)
    lang1_vocab = torchtext.vocab.vocab(
        ordered_dict=lang1_ordered,
        min_freq=min_freq,
        specials=["<pad>", "<unk>", "<sos>", "<eos>"],
        special_first=True,
    )
    lang2_vocab = torchtext.vocab.vocab(
        ordered_dict=lang2_ordered,
        min_freq=min_freq,
        specials=["<pad>", "<unk>", "<sos>", "<eos>"],
        special_first=True,
    )
    lang1_vocab.set_default_index(1)
    lang2_vocab.set_default_index(1)

    # turn sentences into list of indices
    indices1 = [lang1_vocab(s) for s in lang1]
    indices2 = [lang2_vocab(s) for s in lang2]

    # create a dataset and dataloader
    class PairDataset(Dataset):
        def __init__(self, indices1, indices2):
            assert len(indices1) == len(indices2)
            self.indices1 = indices1
            self.indices2 = indices2

        def __len__(self):
            return len(self.indices1)

        def __getitem__(self, idx):
            return torch.tensor(self.indices1[idx], device=device), torch.tensor(
                self.indices2[idx], device=device
            )

    def collate(batch):
        seq1, seq2 = [], []
        for sen1, sen2 in batch:
            seq1.append(sen1)
            seq2.append(sen2)

        pad_seq1 = nn.utils.rnn.pad_sequence(seq1, batch_first=True)
        pad_seq2 = nn.utils.rnn.pad_sequence(seq2, batch_first=True)

        return pad_seq1, pad_seq2

    dataset = PairDataset(indices1, indices2)
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )

    # model
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(
                len(lang1_vocab), enc_hidden_size, padding_idx=0
            )
            self.lstm = nn.LSTM(
                input_size=enc_hidden_size,
                hidden_size=enc_hidden_size,
                num_layers=enc_nlayers,
                batch_first=True,
                bidirectional=True,
            )

        def forward(self, x):
            x = self.embedding(x)
            out, (_, _) = self.lstm(x)
            return out

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(
                len(lang2_vocab), dec_hidden_size, padding_idx=0
            )
            self.lstm = nn.ModuleList(
                [
                    nn.LSTMCell(input_size=dec_hidden_size, hidden_size=dec_hidden_size)
                    for _ in range(dec_nlayers)
                ]
            )
            self.energy_h = nn.Linear(dec_hidden_size, dec_hidden_size)
            self.energy_encoder = nn.Linear(dec_hidden_size, dec_hidden_size)
            self.energy = nn.Linear(dec_hidden_size, 1)

            self.combine_context = nn.Linear(dec_hidden_size, dec_hidden_size)
            self.combine_embedding = nn.Linear(dec_hidden_size, dec_hidden_size)

            self.logits = nn.Linear(dec_hidden_size, len(lang2_vocab))

        def calculate_context(self, h, encoder_outputs):
            # calculate attention
            energy_h = self.energy_h(h).unsqueeze(dim=1)
            energy_encoder = self.energy_encoder(encoder_outputs)
            energy = self.energy(torch.tanh(energy_h + energy_encoder))
            attention = torch.softmax(energy, dim=1).squeeze(2).unsqueeze(1)
            context = (attention @ encoder_outputs).squeeze(1)
            return context

        def forward(self, x, encoder_outputs):
            h = [
                torch.zeros(batch_size, dec_hidden_size, device=device)
                for _ in range(dec_nlayers)
            ]
            c = [
                torch.zeros(batch_size, dec_hidden_size, device=device)
                for _ in range(dec_nlayers)
            ]

            embedding = self.embedding(x)
            _, seq_len, _ = embedding.shape

            outputs = []
            for t in range(seq_len):
                # combine context with embedding
                context = self.calculate_context(h[0], encoder_outputs)
                emb = embedding[:, t, :]
                inp = self.combine_context(context) + self.combine_embedding(emb)

                for i, layer in enumerate(self.lstm):
                    h[i], c[i] = layer(inp, (h[i], c[i]))
                    inp = h[i]
                outputs.append(h[-1])

            outputs = torch.stack(outputs, dim=1)
            logits = self.logits(outputs)
            return logits

        # generate a translation using greedy search
        @torch.inference_mode()
        def decode(self, encoder_outputs):
            indices = []  # list holding the indices for all generated words

            # for simplicity we assume that we translate just one batch of data
            h = [
                torch.zeros(1, dec_hidden_size, device=device)
                for _ in range(dec_nlayers)
            ]
            c = [
                torch.zeros(1, dec_hidden_size, device=device)
                for _ in range(dec_nlayers)
            ]

            # start sentence generation by inputting <sos> token
            last_index = torch.tensor(lang2_vocab(["<sos>"]), device=device)

            # generate sequence, until the <eos> token is generated
            while True:
                context = self.calculate_context(h[0], encoder_outputs)
                emb = self.embedding(last_index)
                inp = self.combine_context(context) + self.combine_embedding(emb)

                for i, layer in enumerate(self.lstm):
                    h[i], c[i] = layer(inp, (h[i], c[i]))
                    inp = h[i]
                logits = self.logits(h[-1])
                last_index = logits.argmax(dim=1)
                if last_index.item() == lang2_vocab["<eos>"] or len(indices) >= max_len:
                    break
                indices.append(last_index.item())
            return indices

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder().to(device)
            self.decoder = Decoder().to(device)

        @torch.inference_mode()
        def translate(self, sentence):
            tokens = tokenize(sentence)
            indices = lang1_vocab(tokens)
            indices = torch.tensor(indices, device=device)
            encodings = self.encoder(indices).unsqueeze(0)  # a batch of 1
            decodings = self.decoder.decode(encodings)
            translation = lang2_vocab.lookup_tokens(decodings)
            return " ".join(translation)

        def forward(self, seq1, seq2):
            encoder_outputs = self.encoder(seq1)
            logits = self.decoder(seq2, encoder_outputs)
            return logits

    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train():
        print("Starting Training")
        # logging variables
        start_time = time.time()
        t0 = start_time

        for epoch in range(num_epochs):
            for batch, (lang1, lang2) in enumerate(dataloader):
                optimizer.zero_grad()

                logits = model(lang1, lang2[:, :-1])
                B, S, C = logits.shape  # batch, sequence, classes
                loss = criterion(logits.reshape(B * S, C), lang2[:, 1:].reshape(B * S))

                loss.backward()
                optimizer.step()

                t1 = time.time()
                time_passed_batch = t1 - t0
                time_passed_start = t1 - start_time
                t0 = t1
                if (batch + 1) % 100 == 0:
                    print(
                        f"Time Since Start {(time_passed_start/ 60):.2f}min, Time Since Last Batch {time_passed_batch:.2f}sec, Epoch: {epoch+1}/{num_epochs}, Batch: {batch+1}/{len(dataloader)}, Train Loss: {loss.item():.4f}"
                    )
        print("Saving Model")
        torch.save(model.state_dict(), f="pretrained.pth")

    if is_training:
        train()
    else:
        # load model and create translation for several german sentences
        print("Loading pretrained model")
        model.load_state_dict(torch.load(f="pretrained.pth"))
        print("Starting Translation")
        sentences = [
            {"original": "I am hungry", "translation": "Ich bin hungrig"},
            {"original": "She is funny", "translation": "Sie ist witzig"},
            {
                "original": "I need a new car!",
                "translation": "Ich brauche ein neues Auto",
            },
            {"original": "It is extremely hot", "translation": "It is extrem heiß"},
            {"original": "I went to college", "translation": "Ich ging zur Uni"},
            {
                "original": "We have many options",
                "translation": "Wir haben viele Möglichkeiten",
            },
            {"original": "Life is not fair", "translation": "Das Leben ist ungerecht"},
            {"original": "To be or not to be", "translation": "Sein oder nicht sein"},
        ]
        for s in sentences:
            print("-" * 50)
            print(f'Generating translation for: {s["original"]}')
            output = model.translate(s["original"])
            print(f'Model Translation: "{output}"')
            print(f'Expected Translation: {s["translation"]}')
