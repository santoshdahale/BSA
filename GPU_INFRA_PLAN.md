# GPU credits and infra plan for SLM training and finetuning

This document provides a practical plan for acquiring GPU credits and running initial SLM/finetuning experiments, cost estimates, and recommended instance types and runtimes.

## Goals
- Small-scale SLM experiments and finetuning of the prototype BSA transformer.
- Run experiments for model sizes up to ~100M params locally; plan for scaling to 1B+ with provisioning.

## Recommended GPUs (budget-minded)
- NVIDIA A10 / A100 40GB (good perf, generally available)
- NVIDIA H100 / H800 for large-scale SLM (higher cost)
- For prototyping: single A100-40GB or A10 is sufficient.

## Cost estimate template (example)
- Small experiment (1 A100-40GB, 4–8 hours): $20–$60
- Medium experiment (1 H100, 24 hours): $200–$400
- Long-run pretraining (multi-GPU, 3–7 days): requires custom quote or cloud reserved instances.

## Practical tips
- Use spot/preemptible instances for cost savings (but add checkpoint frequency and robust resume logic).
- Use DeepSpeed for memory efficiency and ZeRO optimizer shards.
- Offload optimizer state to CPU when possible.
- Use mixed precision (AMP/BF16) to reduce memory and speed up.

## Minimal experiment plan
1. Smoke run on a single GPU (A10/A100): small model (10–50M params), 1-2 epochs on small dataset.
2. Scale to multi-GPU with DeepSpeed ZeRO-2/3 if memory-bound.
3. Run a finetune sweep for programming-type models with distilled teacher/student approach.

## Next steps for procurement
1. Define target model size and dataset size.
2. Estimate total GPU hours needed per experiment.
3. Buy credits from cloud provider or use marketplace credits. I can prepare a cost spreadsheet if you share target model sizes and dataset sizes.
