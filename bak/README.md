# bak/

Archived modules that are **not imported anywhere in the main code path** but are
kept for reference or potential future reactivation.

| File | Purpose | Why archived |
|------|---------|--------------|
| `inference.py` | High-level online-inference wrapper around `RewardModel` with ring-buffer history, FP16, and `torch.compile`. | `sim/sac_train*.py` and `sim/rm_eval.py` implement their own inline window logic; no caller uses `RewardModelInference`. Re-enable if you need a clean single-thread online scorer. |
| `retrieval_reward.py` | Non-parametric KNN reward (cosine similarity to a success-embedding bank). | Ablation baseline that was never wired into training. Re-enable as a no-parametric-RM comparison. |
| `data_structure.txt` | Legacy note on the original data layout. | Historical; `data/data_structure.txt` is the live reference. |

Reactivation checklist:
1. `git mv bak/<file>.py ./` back to the repo root.
2. Add an import site (e.g. a new `scripts/run_*.py`) so the module is no longer orphaned.
3. Run `ruff check` to confirm the module is still lint-clean after any PyTorch/HF API drift.
