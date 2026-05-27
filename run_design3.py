import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import subprocess

EVIDENCE_PATH = "data/evidence.json"
DEV_CLAIMS_PATH = "data/dev-claims.json"
PREDICTION_PATH = "prediction.json"
PREDICTION_PATH_FINAL = "data/prediction_design3.json"

with open(EVIDENCE_PATH, "r", encoding="utf-8") as f:
    evidence = json.load(f)
with open(DEV_CLAIMS_PATH, "r", encoding="utf-8") as f:
    dev_claims = json.load(f)

with open(PREDICTION_PATH, "r", encoding="utf-8") as f:
    predictions = json.load(f)
dev_retrieved = {cid: predictions[cid]["evidences"] for cid in dev_claims.keys()}

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
except Exception:
    MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

model.eval()
predictions_dict = {}
allowed_labels = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]

for cid, cdata in dev_claims.items():
    claim_text = cdata["claim_text"]
    ret_evs = dev_retrieved[cid]
    ev_text = "\n".join([f"- {evidence[eid]}" for eid in ret_evs])
    prompt = f"""System: You are an expert fact-checker. Given a claim and retrieved evidences, classify the claim into exactly one of these labels:
- SUPPORTS: The evidence supports the claim.
- REFUTES: The evidence refutes the claim.
- NOT_ENOUGH_INFO: The evidence does not have enough information to support or refute.
- DISPUTED: Different evidences contradict each other or the claim is highly disputed.

Output your reasoning chain, then output the final label as exactly one of: SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED.

User:
Claim: {claim_text}
Evidence:
{ev_text}

Label:"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    clean_response = response.upper()
    decision_part = clean_response.split("</THOUGHT>")[-1] if "</THOUGHT>" in clean_response else clean_response
    predicted_label = "NOT_ENOUGH_INFO"
    for label in allowed_labels:
        if re.search(r'\b' + label + r'\b', decision_part):
            predicted_label = label
            break
    predictions_dict[cid] = {
        "claim_text": claim_text,
        "claim_label": predicted_label,
        "evidences": ret_evs
    }

with open(PREDICTION_PATH_FINAL, "w", encoding="utf-8") as f:
    json.dump(predictions_dict, f, indent=2)

result = subprocess.run(
    ["python", "eval.py", "--predictions", PREDICTION_PATH_FINAL, "--groundtruth", DEV_CLAIMS_PATH],
    capture_output=True,
    text=True
)
print(result.stdout)
