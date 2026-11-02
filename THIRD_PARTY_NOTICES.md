# Third-party sources

`easy_loss.py` contains these unmodified function bodies from EasyVideoR1:

- `masked_mean` — `verl/utils/torch_functional.py`
- `average_loss`, `compute_policy_loss`, `compute_grpo_outcome_advantage` — `verl/trainer/core_algos.py`

The advantage-registration decorator was omitted to avoid importing the full distributed trainer stack. Imports and a `VF` namespace adapter are provided locally. The caller invokes advantage computation under `torch.no_grad()`.

Source repository: [cyuQ1n/EasyVideoR1](https://github.com/cyuQ1n/EasyVideoR1).

Source checkout commit: `7ab338b33f29e18039e105daa61e9ebabe003c62`.

Source file SHA-256:

```text
verl/utils/torch_functional.py
fd686c86f5e479acb39d873fef8eadc048e616f8c7a868ea50c71e925667e651

verl/trainer/core_algos.py
f059431e21c7d6968901f2c0f762f68d5823588c1f3f9dda15a475b445b81212
```

The source headers identify Apache License 2.0 and copyrights of The HuggingFace Team (2022) and Bytedance Ltd. and/or affiliates (2024). The source repository license is reproduced in [third_party/EasyVideoR1_LICENSE](third_party/EasyVideoR1_LICENSE). See the original headers in `easy_loss.py` as well.

Model weights, datasets, rollouts and optimizer states are not distributed here. Their respective licenses and access conditions remain applicable.
