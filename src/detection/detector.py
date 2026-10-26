"""CIREE fact detector wrapper.

Loads the calibrated RoBERTa model from the Hugging Face Hub (or a local
checkpoint) and labels sentences as factual claims (1) or narrative (0).
Temperature scaling (T = 1.6206) is baked into the weights, so probabilities
come straight from softmax with no extra scaling.
"""
import json
import os
import re

import torch
from huggingface_hub import hf_hub_download
from transformers import RobertaForSequenceClassification, RobertaTokenizer

DEFAULT_MODEL = "ez6047/CIREE-fact-detector"
DEFAULT_THRESHOLD = 0.635  # universal threshold from inference_config.json
THRESHOLDS = {"balanced": 0.597, "new_domain": 0.371, "skewed": 0.896}

SPLIT = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["”]))\s+(?=["“]?[A-Z0-9$])')


def split_sentences(text):
    """Split running text into sentences (CIREE was trained on single sentences)."""
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in SPLIT.split(text) if s.strip()]


def _load_threshold(model):
    if os.path.isdir(model):
        path = os.path.join(model, "inference_config.json")
    else:
        path = hf_hub_download(model, "inference_config.json")
    with open(path) as f:
        return json.load(f)["threshold"]


class CIREEDetector:
    def __init__(self, model=DEFAULT_MODEL, threshold=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        self.model = RobertaForSequenceClassification.from_pretrained(model).to(self.device).eval()
        if threshold is None:
            try:
                threshold = _load_threshold(model)
            except Exception:
                threshold = DEFAULT_THRESHOLD
        self.threshold = THRESHOLDS.get(threshold, threshold)

    @torch.no_grad()
    def score(self, sentences, batch_size=16):
        """Return the calibrated fact probability for each sentence."""
        probs = []
        for i in range(0, len(sentences), batch_size):
            enc = self.tokenizer(sentences[i:i + batch_size], truncation=True, padding=True,
                                 max_length=256, return_tensors="pt").to(self.device)
            probs += torch.softmax(self.model(**enc).logits, -1)[:, 1].tolist()
        return probs

    def predict(self, sentences, batch_size=16):
        """Return [{sentence, fact_probability, label}] for a list of sentences."""
        probs = self.score(sentences, batch_size)
        return [{"sentence": s, "fact_probability": p, "label": int(p >= self.threshold)}
                for s, p in zip(sentences, probs)]

    def detect(self, text, batch_size=16):
        """Split text into sentences and return only the ones labelled as facts."""
        return [r for r in self.predict(split_sentences(text), batch_size) if r["label"]]
