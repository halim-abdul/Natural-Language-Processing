import numpy as np
import torch
from scipy.stats import pearsonr

@torch.no_grad()
def evaluate_sst(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    for batch in dataloader:
        logits = model.predict_sentiment(
            batch["token_ids"].to(device),
            batch["attention_mask"].to(device),
        )
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return float(np.mean(np.array(preds) == np.array(labels))) if labels else 0.0

@torch.no_grad()
def evaluate_qqp(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    for batch in dataloader:
        logits = model.predict_paraphrase(
            batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device),
            batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device),
        )
        preds.extend((torch.sigmoid(logits) >= 0.5).long().cpu().tolist())
        labels.extend(batch["labels"].long().cpu().tolist())
    return float(np.mean(np.array(preds) == np.array(labels))) if labels else 0.0

@torch.no_grad()
def evaluate_sts(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    for batch in dataloader:
        scores = model.predict_similarity(
            batch["token_ids_1"].to(device), batch["attention_mask_1"].to(device),
            batch["token_ids_2"].to(device), batch["attention_mask_2"].to(device),
        )
        preds.extend(scores.cpu().tolist())
        labels.extend(batch["labels"].float().cpu().tolist())
    if len(labels) < 2:
        return 0.0
    return float(pearsonr(preds, labels).statistic)

def model_eval_multitask(sst_dataloader, para_dataloader, sts_dataloader, model, device):
    sst_acc = evaluate_sst(model, sst_dataloader, device) if sst_dataloader else 0.0
    para_acc = evaluate_qqp(model, para_dataloader, device) if para_dataloader else 0.0
    sts_corr = evaluate_sts(model, sts_dataloader, device) if sts_dataloader else 0.0
    return sst_acc, [], [], para_acc, [], [], sts_corr, [], []

def test_model_multitask(*args, **kwargs):
    raise NotImplementedError("Use the task-specific prediction methods in multitask_classifier.py.")
