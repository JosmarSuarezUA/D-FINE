#!/usr/bin/env python
"""
evaluate_dfine.py
=================
Cross-domain evaluation of D-FINE: every source checkpoint is evaluated on
the test split of every dataset. The protocol, metrics and adapter live in the
shared ``da-eval`` package (https://github.com/JosmarSuarezUA/da-eval).

- Datasets are defined once in ``da-eval/configs/datasets.yaml`` (shared by all models).
- ``CHECKPOINTS`` maps each source dataset to the checkpoint trained on it.
- Model settings (config, batch size, device, ...) go to ``DFINEAdapter``.

Equivalent CLI::

    uv run da-eval --model dfine \\
        --config configs/dfine/custom/objects365/dfine_hgnetv2_s_obj2custom.yml \\
        --datasets ../da-eval/configs/datasets.yaml \\
        --checkpoint A=output/<sds_run>/best_stg1.pth \\
        --result-root dfine_results --wandb-project dfine
"""

from pathlib import Path

from da_eval import load_dataset_configs, run_all_sources
from da_eval.adapters.yaml_engine import DFINEAdapter

DATASETS = Path(__file__).resolve().parent.parent / "da-eval" / "configs" / "datasets.yaml"

# TODO: set the checkpoint trained on each source dataset (keys match datasets.yaml).
CHECKPOINTS = {
    "A": "output/<sds_run>/best_stg1.pth",
    "B": "output/<synbase_run>/best_stg1.pth",
    "C": "output/<afo_run>/best_stg1.pth",
}

if __name__ == "__main__":
    adapter = DFINEAdapter(config_path="configs/dfine/custom/objects365/dfine_hgnetv2_s_obj2custom.yml")
    run_all_sources(
        adapter,
        checkpoints=CHECKPOINTS,
        dataset_configs=load_dataset_configs(DATASETS),
        result_root="dfine_results",
        wandb_project="dfine",   # optional
    )
