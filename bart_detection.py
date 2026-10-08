import argparse
import ast
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BartModel

DROP_IDS = {12, 19, 20, 23, 27}
VALID_TYPES = [i for i in range(1, 32) if i not in DROP_IDS]
TYPE_TO_INDEX = {t: i for i, t in enumerate(VALID_TYPES)}

class ETPCDetectionDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=256):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        enc = self.tokenizer(
            str(row["sentence1"]), str(row["sentence2"]),
            max_length=self.max_length, truncation=True, padding="max_length",
            return_tensors="pt",
        )
        labels = torch.zeros(len(VALID_TYPES))
        raw = row.get("paraphrase_type_ids", "[]")
        try:
            ids = ast.literal_eval(raw) if isinstance(raw, str) else raw
        except Exception:
            ids = []
        for t in ids:
            if int(t) in TYPE_TO_INDEX:
                labels[TYPE_TO_INDEX[int(t)]] = 1.0
        return {k: v.squeeze(0) for k, v in enc.items()}, labels

class BartTypeDetector(nn.Module):
    def __init__(self, model_name="facebook/bart-large", dropout=0.2):
        super().__init__()
        self.bart = BartModel.from_pretrained(model_name)
        h = self.bart.config.d_model
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, h//2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(h//2, len(VALID_TYPES)))

    def forward(self, **batch):
        out = self.bart(**batch)
        pooled = out.last_hidden_state[:, 0]
        return self.head(pooled)

def train(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    df = pd.read_csv(args.train_file)
    train_df, dev_df = train_test_split(df, test_size=0.2, random_state=args.seed)

    train_loader = DataLoader(ETPCDetectionDataset(train_df, tokenizer), batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(ETPCDetectionDataset(dev_df, tokenizer), batch_size=args.batch_size)

    model = BartTypeDetector(args.model_name).to(device)
    positives = torch.zeros(len(VALID_TYPES))
    negatives = torch.zeros(len(VALID_TYPES))
    for _, y in train_loader:
        positives += y.sum(0)
        negatives += y.size(0) - y.sum(0)
    pos_weight = (negatives / positives.clamp_min(1)).clamp(0.2, 5.0).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(args.epochs):
        model.train()
        for batch, y in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(**batch), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        probs, gold = [], []
        with torch.no_grad():
            for batch, y in dev_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                probs.append(torch.sigmoid(model(**batch)).cpu())
                gold.append(y)
        probs, gold = torch.cat(probs), torch.cat(gold)
        pred = (probs >= 0.5).int().numpy().ravel()
        g = gold.int().numpy().ravel()
        print(f"epoch={epoch+1} dev_mcc={matthews_corrcoef(g, pred):.4f}")

    torch.save(model.state_dict(), args.save_path)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default="data/etpc-paraphrase-train.csv")
    p.add_argument("--model_name", default="facebook/bart-large")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=11711)
    p.add_argument("--save_path", default="bart_detection.pt")
    p.add_argument("--use_gpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    train(get_args())
