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
PREDICTION_PATH_FINAL = "data/prediction_design1.json"

with open(EVIDENCE_PATH, "r", encoding="utf-8") as f:
    evidence = json.load(f)
with open(DEV_CLAIMS_PATH, "r", encoding="utf-8") as f:
    dev_claims = json.load(f)
with open(TRAIN_CLAIMS_PATH, "r", encoding="utf-8") as f:
    train_claims = json.load(f)

with open(PREDICTION_PATH, "r", encoding="utf-8") as f:
    predictions = json.load(f)
dev_retrieved = {cid: predictions[cid]["evidences"] for cid in dev_claims.keys()}

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 512
label_map = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
inv_label_map = {v: k for k, v in label_map.items()}

class ClaimEvidenceDataset(Dataset):
    def __init__(self, claims_dict, retrieved_dict, evidence_dict, tokenizer, max_len, use_gold=False):
        self.items = []
        for cid, cdata in claims_dict.items():
            claim_text = cdata["claim_text"]
            ev_ids = cdata["evidences"] if use_gold else retrieved_dict[cid]
            ev_text = " ".join([evidence_dict[eid] for eid in ev_ids])
            label = label_map[cdata["claim_label"]]
            self.items.append((claim_text, ev_text, label, cid))
        claims_list = [item[0] for item in self.items]
        evidence_list = [item[1] for item in self.items]
        self.encodings = tokenizer(
            claims_list,
            evidence_list,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors=None
        )
        self.labels = [item[2] for item in self.items]
        self.claim_ids = [item[3] for item in self.items]
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = ClaimEvidenceDataset(train_claims, None, evidence, tokenizer, MAX_LEN, use_gold=True)
dev_dataset = ClaimEvidenceDataset(dev_claims, dev_retrieved, evidence, tokenizer, MAX_LEN, use_gold=False)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./results_design1",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="no",
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
