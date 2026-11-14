# Reproducibility audit

Both experiments use `outputs/splits/dataset_splits.json`, the same transformations, optimization settings, and evaluation implementation. Test IDs are never loaded by training.

| Model | Max epochs | Actual epochs | Optimizer updates | Best validation epoch | Train seconds | Parameters | Inference seconds | Hardware |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 20 | 14 | 1456 | 10 | 609.7382617001422 | 1927042 | 75.33606439991854 | cuda:NVIDIA GeForce GTX 1650 |
| zeroed_skip | 20 | 8 | 832 | 4 | 436.4262753999792 | 1927042 | 40.487228699959815 | cuda:NVIDIA GeForce GTX 1650 |

Training duration difference (zeroed_skip - baseline): -173.312 seconds.

The zeroed-skip model is a controlled zeroed-skip-fusion ablation: it uses the exact same encoder features, decoder blocks, concatenation widths, and trainable parameters, but replaces each skip tensor with `torch.zeros_like`.
Checkpoints store their configuration and seed. Evaluation metrics are computed only over valid, non-ignored pixels; ignore value 255 is not a segmentation class.
