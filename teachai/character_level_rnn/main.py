import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical


class Model(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=args.dropout,
        )
        self.logits = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        outputs, (_, _) = self.rnn(x)
        logits = self.logits(outputs)
        return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Character Level LSTM")
    parser.add_argument(
        "--exp-name",
        type=str,
        default="character_level_lstm",
        help="name of the experiment for logging and tracking",
    )
    parser.add_argument(
        "--input", type=str, default="input.txt", help="input file used for training"
    )
    parser.add_argument(
        "--train",
        type=bool,
        default=False,
        help="train the model and overwrite existing pretrained model",
    )
    # hyperparameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="the device on which training and inference will run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default="42",
        help="the seed for deterministic results",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="number of epochs used for training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default="1024",
        help="batch size used in training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default="4",
        help="number of workers used in data loaders",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=100,
        help="length of the context window for RNN",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="size of hidden vector in lstm",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="number of layers in lstm",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout percentage",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Preparation
    # -----------------------------------------------------------------
    torch.manual_seed(args.seed)

    # -----------------------------------------------------------------
    # Prepare Data
    # -----------------------------------------------------------------
    with open(args.input, mode="r") as f:
        text = f.read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    data = torch.tensor(encode(text))
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]

    class CharDataset(Dataset):
        def __init__(self, data):
            super().__init__()
            self.data = data

        def __len__(self):
            return len(self.data) - args.context_size

        def __getitem__(self, idx):
            x = self.data[idx : idx + args.context_size]
            y = self.data[idx + 1 : idx + args.context_size + 1]
            return x, y

    train_dataset = CharDataset(train_data)
    test_dataset = CharDataset(test_data)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    # -----------------------------------------------------------------
    # Evaluation Function
    # -----------------------------------------------------------------
    @torch.inference_mode()
    def evaluate(dataloader):
        model.eval()
        losses = []
        for x, y in dataloader:
            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            B, S, C = logits.shape  # batch, sequence, classes
            loss = F.cross_entropy(logits.view(B * S, C), y.view(B * S))
            losses.append(loss.item())
        model.train()
        return torch.tensor(losses).mean().item()

    # -----------------------------------------------------------------
    # Sample Text
    # -----------------------------------------------------------------
    @torch.inference_mode()
    def generate_sample(start_idx=0, output_len=100):
        sequence = []
        sequence.append(start_idx)
        model.eval()
        for _ in range(output_len):
            inp = sequence
            if len(inp) > args.context_size:
                inp = inp[-args.context_size :]
            inp = torch.tensor(inp, device=args.device)
            logits = model(inp)[-1]
            distribution = Categorical(logits=logits)
            idx = distribution.sample().item()
            sequence.append(idx)
        model.train()
        return decode(sequence)

    # -----------------------------------------------------------------
    # Model, Optimizer
    # -----------------------------------------------------------------
    model = Model(
        vocab_size=vocab_size, hidden_size=args.hidden_size, num_layers=args.num_layers
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )

    # -----------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------

    def train():
        # logging variables
        start_time = time.time()
        t0 = start_time
        best_loss = float("inf")

        for epoch in range(args.num_epochs):
            for batch, (x, y) in enumerate(train_dataloader):
                optimizer.zero_grad()

                x = x.to(args.device)
                y = y.to(args.device)

                logits = model(x)
                B, S, C = logits.shape  # batch, sequence, classes
                loss = F.cross_entropy(logits.view(B * S, C), y.view(B * S))

                loss.backward()
                optimizer.step()
                val_loss = evaluate(test_dataloader)
                scheduler.step(val_loss)

                t1 = time.time()
                time_passed_batch = t1 - t0
                time_passed_start = t1 - start_time
                t0 = t1
                print(
                    f"Time Since Start {(time_passed_start/ 60):.2f}min, Time Since Last Batch {time_passed_batch:.2f}sec, Epoch: {epoch+1}/{args.num_epochs}, Batch: {batch+1}/{len(train_dataloader)}, Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}"
                )

                # save weights
                if val_loss < best_loss:
                    print("Saving Weights")
                    best_loss = val_loss
                    torch.save(model.state_dict(), f="pretrained.pth")
                print("-" * 100)

                # sample a sentence from time to time
                if (batch + 1) % 10 == 0:
                    print(generate_sample(output_len=100))

    if args.train:
        train()

    # -----------------------------------------------------------------
    # Sample Text
    # -----------------------------------------------------------------
    with open(f"sample.txt", mode="w+") as f:
        model.load_state_dict(torch.load("pretrained.pth"))
        sample = generate_sample(output_len=10000)
        f.write(sample)
