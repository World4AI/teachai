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
min_freq = 5  # the minimum amount of times a word has to occur in order to be included
batch_size = 512
max_len = 10
embedding_dim = 256
head_dim = 64
fc_dim = embedding_dim * 4
num_heads = embedding_dim // head_dim
num_layers = 2
dropout = 0.1
num_epochs = 10
lr = 0.0001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_training = False  # train or evaluate


class Embeddings(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.position_embedding = nn.Embedding(max_len, embedding_dim)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(
            torch.arange(0, seq_len, device=device).view(1, seq_len)
        )
        return token_embedding + position_embedding


def attention(query, key, value, mask=None):
    scores = (query @ key.transpose(1, 2)) / torch.tensor(
        embedding_dim, device=device
    ).sqrt()
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = torch.softmax(scores, -1)
    attention = attn_weights @ value
    return attention


class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_dim)
        self.key = nn.Linear(embedding_dim, head_dim)
        self.value = nn.Linear(embedding_dim, head_dim)

    def forward(self, query, key, value, mask=None):
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        return attention(query, key, value, mask)


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead() for _ in range(num_heads)])
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        x = [head(query, key, value, mask) for head in self.heads]
        x = torch.cat(x, dim=-1)
        x = self.dropout(self.output(x))
        return x


class PWFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        return self.layers(x)


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.self_attention = MultiHeadAttention()
        self.feed_forward = PWFeedForward()

    def forward(self, src, mask=None):
        normalized = self.norm1(src)
        src = src + self.self_attention(normalized, normalized, normalized, mask)
        normalized = self.norm2(src)
        src = src + self.feed_forward(normalized)
        return src


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])

    def forward(self, src, mask=None):
        for encoder in self.layers:
            src = encoder(src, mask)
        return src


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.self_attention = MultiHeadAttention()
        self.cross_attention = MultiHeadAttention()
        self.feed_forward = PWFeedForward()

    def forward(self, src, trg, src_mask, trg_mask):
        normalized = self.norm1(trg)
        trg = trg + self.self_attention(normalized, normalized, normalized, trg_mask)
        normalized = self.norm2(trg)
        trg = trg + self.cross_attention(trg, src, src, src_mask)
        normalized = self.norm3(trg)
        trg = trg + self.feed_forward(normalized)
        return trg


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        for decoder in self.layers:
            trg = decoder(src, trg, src_mask, trg_mask)
        return trg


class Transformer(nn.Module):
    def __init__(self, vocab_size_src, vocab_size_trg):
        super().__init__()
        self.src_embedding = Embeddings(vocab_size=vocab_size_src)
        self.trg_embedding = Embeddings(vocab_size=vocab_size_trg)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.logits = nn.Linear(embedding_dim, vocab_size_trg)

    def forward(self, src, trg):
        src_mask = (src != 0).to(torch.int64).unsqueeze(1)
        trg_mask = torch.tril(torch.ones(trg.shape[1], trg.shape[1], device=device))
        src = self.src_embedding(src)
        trg = self.trg_embedding(trg)

        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(encoder_output, trg, src_mask, trg_mask)
        logits = self.logits(decoder_output)
        return logits

    # generate a translation using greedy search
    @torch.inference_mode()
    def translate(self, sentence):
        tokens = tokenize(sentence)
        src_indices = src_vocab(tokens)
        src_indices = torch.tensor(src_indices, device=device).unsqueeze(0)
        src = self.src_embedding(src_indices)
        encoder_output = self.encoder(src)

        trg_indices = []
        # start sentence generation by inputting <sos> token
        trg_indices.append(trg_vocab["<sos>"])
        while True:
            trg_input = torch.tensor(trg_indices, device=device).unsqueeze(0)
            trg = self.trg_embedding(trg_input)
            decoder_output = self.decoder(encoder_output, trg)
            logits = self.logits(decoder_output)
            logits = logits[:, -1, :]
            last_index = logits.argmax(dim=1)
            if last_index.item() == trg_vocab["<eos>"] or len(trg_indices) >= max_len:
                break
            trg_indices.append(last_index.item())
        translation = trg_vocab.lookup_tokens(trg_indices[1:])
        return " ".join(translation)


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
    src, trg = [], []
    with open(path, "r") as f:
        for line in f:
            # the third element is the licence
            sen_1, sen_2, _ = line.split("\t")
            sen_1, sen_2 = tokenize(sen_1), tokenize(sen_2)

            if len(sen_1) <= max_len and len(sen_2) <= max_len:
                src.append(sen_1)
                trg.append(sen_2)

    # create vocabulary
    src_counter = Counter()
    trg_counter = Counter()

    for sen_1, sen_2 in zip(src, trg):
        src_counter.update(sen_1)
        trg_counter.update(sen_2)

    src_sorted = sorted(src_counter.items(), key=lambda x: x[1], reverse=True)
    trg_sorted = sorted(trg_counter.items(), key=lambda x: x[1], reverse=True)
    src_ordered = OrderedDict(src_sorted)
    trg_ordered = OrderedDict(trg_sorted)
    src_vocab = torchtext.vocab.vocab(
        ordered_dict=src_ordered,
        min_freq=min_freq,
        specials=["<pad>", "<unk>", "<sos>", "<eos>"],
        special_first=True,
    )
    trg_vocab = torchtext.vocab.vocab(
        ordered_dict=trg_ordered,
        min_freq=min_freq,
        specials=["<pad>", "<unk>", "<sos>", "<eos>"],
        special_first=True,
    )
    src_vocab.set_default_index(1)
    trg_vocab.set_default_index(1)

    # turn sentences into list of indices
    src_indices = [src_vocab(s) for s in src]
    trg_indices = [trg_vocab(s) for s in trg]

    # create a dataset and dataloader
    class PairDataset(Dataset):
        def __init__(self, src_indices, trg_indices):
            assert len(src_indices) == len(trg_indices)
            self.src_indices = src_indices
            self.trg_indices = trg_indices

        def __len__(self):
            return len(self.src_indices)

        def __getitem__(self, idx):
            return torch.tensor(self.src_indices[idx], device=device), torch.tensor(
                self.trg_indices[idx], device=device
            )

    def collate(batch):
        src_seq, trg_seq = [], []
        for src_sentence, trg_sentece in batch:
            src_seq.append(src_sentence)
            trg_seq.append(trg_sentece)

        src_pad = nn.utils.rnn.pad_sequence(src_seq, batch_first=True)
        trg_pad = nn.utils.rnn.pad_sequence(trg_seq, batch_first=True)

        return src_pad, trg_pad

    dataset = PairDataset(src_indices, trg_indices)
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
    )

    model = Transformer(len(src_vocab), len(trg_vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    def train():
        print("Starting Training")
        # logging variables
        start_time = time.time()
        t0 = start_time

        for epoch in range(num_epochs):
            for batch, (src, trg) in enumerate(dataloader):
                optimizer.zero_grad()

                logits = model(src, trg[:, :-1])
                B, S, C = logits.shape  # batch, sequence, classes
                loss = criterion(logits.reshape(B * S, C), trg[:, 1:].reshape(B * S))

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
