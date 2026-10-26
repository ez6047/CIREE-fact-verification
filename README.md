# CIREE Fact Verification

End-to-end fact verification for nature documentary transcripts, built on the
[CIREE fact detector](https://huggingface.co/ez6047/CIREE-fact-detector)
(fine-tuned RoBERTa-base, calibrated, F1 0.82 mean across three test sets).

## Pipeline

| Stage | Module | Status |
|---|---|---|
| 1. Detection: flag sentences that are factual claims | `src/detection/` | Wrapper ready |
| 2. Normalization: rewrite claims to be self-contained and checkable | `src/normalization/` | Week 1 |
| 3. Retrieval: find evidence for each claim | `src/retrieval/` | Week 2+ |
| 4. Verification: judge claims against evidence | `src/verification/` | Week 3+ |

## Layout

```
src/            pipeline stages (one package each)
data/raw/       source transcripts (not committed)
data/processed/ intermediate outputs (not committed)
notebooks/      Databricks-exportable .ipynb
tests/          pytest suite
```

## Setup

```bash
pip install -r requirements.txt
pytest -m "not slow"   # fast tests; drop the filter to also load the model
```

## Detection usage

```python
import sys; sys.path.insert(0, "src")
from detection import CIREEDetector

det = CIREEDetector()                       # universal threshold 0.635
det = CIREEDetector(threshold="new_domain") # 0.371 for unseen domains
det.detect("In the last 50 years, wildlife populations have declined by 60 percent. We can change it.")
```

Thresholds: `balanced` 0.597, `new_domain` 0.371, `skewed` 0.896, or pass any float.
The model's temperature (T = 1.6206) is baked into the weights; do not rescale.

On Databricks, add the repo's `src/` folder to `sys.path` the same way, or install requirements on the cluster.
