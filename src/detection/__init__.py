"""Fact detection: wrapper around the CIREE fact detector (huggingface.co/ez6047/CIREE-fact-detector)."""
from .detector import CIREEDetector, split_sentences

__all__ = ["CIREEDetector", "split_sentences"]
