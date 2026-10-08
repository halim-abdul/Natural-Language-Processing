import argparse
import ast
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BartForConditionalGeneration

from paraphrase_type_generation_helpers import _type_aware_augment, _mild_random_mask

class ETPCGenerationDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=192, augment=False):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        source = str(row["sentence1"])
        target = str(row["sentence2"])
        types = str(row.get("paraphrase_type_ids", "[]"))
        segs = str(row.get("sentence1_segment_location", "[]"))

        if self.augment:
            r = random.random()
            if r < 0.4:
                source = _type_aware_augment(source, segs, types, self.tokenizer, p_edit=0.25)
            elif r < 0.5:
                source = _mild_random_mask(source, self.tokenizer)

        prompt = f"types={types} | {source}"
        x = self.tokenizer(prompt, max_length=self.max_length, truncation=True,
                           padding="max_length", return_tensors="pt")
        y = self.tokenizer(target, max_length=self.max_length, truncation=True,
                           padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
        y[y == self.tokenizer.pad_token_id] = -100
        return {k: v.squeeze(0) for k, v in x.items()}, y

def train(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name).to(device)
    df = pd.read_csv(args.train_file)
    loader = DataLoader(ETPCGenerationDataset(df, tokenizer, augment=args.augment),
                        batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch, labels in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(**batch, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += out.loss.item()
        print(f"epoch={epoch+1} train_loss={total/max(1,len(loader)):.4f}")

    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

def generate(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.save_dir)
    model = BartForConditionalGeneration.from_pretrained(args.save_dir).to(device)
    df = pd.read_csv(args.test_file)
    outputs = []
    for _, row in df.iterrows():
        types = str(row.get("paraphrase_type_ids", "[]"))
        prompt = f"types={types} | {row['sentence1']}"
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        ids = model.generate(**enc, max_length=args.max_length, num_beams=args.num_beams,
                             no_repeat_ngram_size=3, early_stopping=True)
        outputs.append(tokenizer.decode(ids[0], skip_special_tokens=True))
    pd.DataFrame({"id": df["id"], "generated": outputs}).to_csv(args.output_file, index=False)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "generate"], default="train")
    p.add_argument("--train_file", default="data/etpc-paraphrase-train.csv")
    p.add_argument("--test_file", default="data/etpc-paraphrase-generation-test-student.csv")
    p.add_argument("--output_file", default="predictions/bart/etpc-paraphrase-generation-test-output.csv")
    p.add_argument("--model_name", default="facebook/bart-large")
    p.add_argument("--save_dir", default="bart_generation_model")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=192)
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--use_gpu", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args) if args.mode == "train" else generate(args)
