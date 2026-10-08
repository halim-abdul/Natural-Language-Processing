import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import evaluate_sst, evaluate_qqp, evaluate_sts
from optimizer import AdamW

class MultitaskBERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=args.local_files_only
        )
        h = self.bert.config.hidden_size
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.sentiment_classifier = nn.Linear(h, 5)
        self.paraphrase_classifier = nn.Linear(h * 4, 1)
        self.similarity_regressor = nn.Sequential(
            nn.Linear(h * 4, h),
            nn.GELU(),
            nn.Dropout(args.hidden_dropout_prob),
            nn.Linear(h, 1),
        )

    def encode(self, ids, mask):
        out = self.bert(ids, mask)
        hidden = out["last_hidden_state"]
        cls = out["pooler_output"]
        maskf = mask.unsqueeze(-1).float()
        mean = (hidden * maskf).sum(1) / maskf.sum(1).clamp_min(1.0)
        return cls, mean

    def pair_features(self, ids1, mask1, ids2, mask2):
        c1, m1 = self.encode(ids1, mask1)
        c2, m2 = self.encode(ids2, mask2)
        a = 0.5 * (c1 + c2)
        b = torch.abs(c1 - c2)
        c = 0.5 * (m1 + m2)
        d = torch.abs(m1 - m2)
        return torch.cat([a, b, c, d], dim=-1)

    def predict_sentiment(self, ids, mask):
        cls, _ = self.encode(ids, mask)
        return self.sentiment_classifier(self.dropout(cls))

    def predict_paraphrase(self, ids1, mask1, ids2, mask2):
        return self.paraphrase_classifier(self.dropout(self.pair_features(ids1, mask1, ids2, mask2))).squeeze(-1)

    def predict_similarity(self, ids1, mask1, ids2, mask2):
        return self.similarity_regressor(self.dropout(self.pair_features(ids1, mask1, ids2, mask2))).squeeze(-1)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")

    train_data = load_multitask_data(args.sst_train, args.qqp_train, args.sts_train, args.etpc_train, split="train")
    dev_data = load_multitask_data(args.sst_dev, args.qqp_dev, args.sts_dev, args.etpc_train, split="dev")
    sst_train, _, qqp_train, sts_train, _ = train_data
    sst_dev, _, qqp_dev, sts_dev, _ = dev_data

    loaders = {
        "sst": DataLoader(SentenceClassificationDataset(sst_train, args), batch_size=args.batch_size, shuffle=True,
                          collate_fn=SentenceClassificationDataset(sst_train, args).collate_fn),
        "qqp": DataLoader(SentencePairDataset(qqp_train, args), batch_size=args.batch_size, shuffle=True,
                          collate_fn=SentencePairDataset(qqp_train, args).collate_fn),
        "sts": DataLoader(SentencePairDataset(sts_train, args, isRegression=True), batch_size=args.batch_size, shuffle=True,
                          collate_fn=SentencePairDataset(sts_train, args, isRegression=True).collate_fn),
    }
    dev_loaders = {
        "sst": DataLoader(SentenceClassificationDataset(sst_dev, args), batch_size=args.batch_size,
                          collate_fn=SentenceClassificationDataset(sst_dev, args).collate_fn),
        "qqp": DataLoader(SentencePairDataset(qqp_dev, args), batch_size=args.batch_size,
                          collate_fn=SentencePairDataset(qqp_dev, args).collate_fn),
        "sts": DataLoader(SentencePairDataset(sts_dev, args, isRegression=True), batch_size=args.batch_size,
                          collate_fn=SentencePairDataset(sts_dev, args, isRegression=True).collate_fn),
    }

    model = MultitaskBERT(args).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        for batch in loaders[args.task]:
            optimizer.zero_grad()
            if args.task == "sst":
                logits = model.predict_sentiment(batch["token_ids"].to(device), batch["attention_mask"].to(device))
                loss = F.cross_entropy(logits, batch["labels"].to(device), label_smoothing=args.label_smoothing)
            elif args.task == "qqp":
                logits = model.predict_paraphrase(
                    batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device),
                    batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device),
                )
                loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].float().to(device))
            else:
                pred = model.predict_similarity(
                    batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device),
                    batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device),
                )
                loss = F.mse_loss(pred, batch["labels"].float().to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metric = {
            "sst": evaluate_sst,
            "qqp": evaluate_qqp,
            "sts": evaluate_sts,
        }[args.task](model, dev_loaders[args.task], device)
        print(f"epoch={epoch+1} task={args.task} dev_metric={metric:.4f}")

    torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["sst", "qqp", "sts"], default="sst")
    p.add_argument("--option", choices=["pretrain", "finetune"], default="finetune")
    p.add_argument("--use_gpu", action="store_true")
    p.add_argument("--local_files_only", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=11711)
    p.add_argument("--save_path", default="multitask_model.pt")
    p.add_argument("--sst_train", default="data/sst-sentiment-train.csv")
    p.add_argument("--sst_dev", default="data/sst-sentiment-dev.csv")
    p.add_argument("--qqp_train", default="data/quora-paraphrase-train.csv")
    p.add_argument("--qqp_dev", default="data/quora-paraphrase-dev.csv")
    p.add_argument("--sts_train", default="data/sts-similarity-train.csv")
    p.add_argument("--sts_dev", default="data/sts-similarity-dev.csv")
    p.add_argument("--etpc_train", default="data/etpc-paraphrase-train.csv")
    return p.parse_args()

if __name__ == "__main__":
    train(get_args())
