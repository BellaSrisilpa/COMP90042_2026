import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import subprocess

EVIDENCE_PATH = "data/evidence.json"
DEV_CLAIMS_PATH = "data/dev-claims.json"
TRAIN_CLAIMS_PATH = "data/train-claims.json"
PREDICTION_PATH = "prediction.json"
PREDICTION_PATH_FINAL = "data/prediction_design2.json"

with open(EVIDENCE_PATH, "r", encoding="utf-8") as f:
    evidence = json.load(f)
with open(DEV_CLAIMS_PATH, "r", encoding="utf-8") as f:
    dev_claims = json.load(f)
with open(TRAIN_CLAIMS_PATH, "r", encoding="utf-8") as f:
    train_claims = json.load(f)

with open(PREDICTION_PATH, "r", encoding="utf-8") as f:
    predictions = json.load(f)
dev_retrieved = {cid: predictions[cid]["evidences"] for cid in dev_claims.keys()}

label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
inv_label_map = {v: k for k, v in label_map.items()}

train_texts = [f"Claim: {cdata['claim_text']} Evidence: {' '.join([evidence[eid] for eid in cdata['evidences']])}" for cdata in train_claims.values()]
train_labels = [label_map[cdata["claim_label"]] for cdata in train_claims.values()]

dev_texts = [f"Claim: {cdata['claim_text']} Evidence: {' '.join([evidence[eid] for eid in dev_retrieved[cid]])}" for cid, cdata in dev_claims.items()]
dev_labels = [label_map[cdata["claim_label"]] for cdata in dev_claims.values()]

MODEL_NAME = "microsoft/deberta-v3-large"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class FactCheckingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len, return_tensors=None)
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = FactCheckingDataset(train_texts, train_labels, tokenizer)
dev_dataset = FactCheckingDataset(dev_texts, dev_labels, tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

training_args = TrainingArguments(
    output_dir="./results_design2",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_ratio=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    fp16=True,
    logging_steps=1000,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=lambda p: {"accuracy": np.mean(np.argmax(p.predictions, axis=1) == p.label_ids)}
)
trainer.train()

predictions = trainer.predict(dev_dataset)
dev_preds = np.argmax(predictions.predictions, axis=1)

predictions_dict = {}
for i, (cid, cdata) in enumerate(dev_claims.items()):
    predictions_dict[cid] = {
        "claim_text": cdata["claim_text"],
        "claim_label": inv_label_map[dev_preds[i]],
        "evidences": dev_retrieved[cid]
    }

with open(PREDICTION_PATH_FINAL, "w", encoding="utf-8") as f:
    json.dump(predictions_dict, f, indent=2)

result = subprocess.run(
    ["python", "eval.py", "--predictions", PREDICTION_PATH_FINAL, "--groundtruth", DEV_CLAIMS_PATH],
    capture_output=True,
    text=True
)
print(result.stdout)
