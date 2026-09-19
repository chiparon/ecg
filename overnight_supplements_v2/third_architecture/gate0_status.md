# Third-architecture gate 0

DEFERRED — inventory only; no model execution performed.

- Public candidate: [tsai InceptionTime](https://github.com/timeseriesAI/tsai/blob/bbb61982741d466bfa81669edc0e17f1971980af/tsai/models/InceptionTime.py), commit `bbb61982741d466bfa81669edc0e17f1971980af`; Apache-2.0. This is the repository author's unofficial PyTorch implementation, not a claim of official InceptionTime implementation.
- Planned input: float32 batch × 12 × 1000, 100 Hz; 5 multilabel logits. Six modules, 32 filters per branch, effective kernels 39/19/9, residual links every three modules, global average pooling. Static source inspection only.
- Planned split: reuse the exact phase2 train/validation/test identities recorded in the manifest; no new split or training.
- GPU inventory: `NVIDIA GeForce RTX 4070 Laptop GPU, 581.08, 8188 MiB`. Device listing is not model execution or a CUDA compatibility test.
- Data paths and immutable public-source hashes are in `gate0_manifest.json`. Remote GPU was not checked or used for this CPU-only reanalysis.
- Dependency/model import, forward/backward, GPU smoke, memory/epoch timing, and training remain unverified and require a separately approved task.

training_invocations = 0
inference_invocations = 0
model_imports = 0
