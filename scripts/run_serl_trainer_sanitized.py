#!/usr/bin/env python3
"""Run the external ManiSkill SERL trainer without its embedded W&B credential.

The external trainer currently contains a committed API key and does not print
evaluation metrics locally. This launcher removes that assignment in memory
and emits every evaluation payload as ``SERL_EVAL_JSON`` in the actor log.
It does not modify the external SERL checkout.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trainer", type=Path, required=True)
    known, trainer_args = parser.parse_known_args()

    source = known.trainer.read_text()
    source, credential_count = re.subn(
        r'^os\.environ\["WANDB_API_KEY"\]\s*=\s*["\'][^"\']+["\']\s*$',
        "",
        source,
        flags=re.MULTILINE,
    )
    if credential_count != 1:
        raise RuntimeError(
            f"Expected exactly one embedded WANDB_API_KEY assignment in {known.trainer}, "
            f"found {credential_count}"
        )

    anchor = '            stats = {"eval": evaluate_info}\n'
    replacement = (
        '            print("SERL_EVAL_JSON " + __import__("json").dumps('
        '{"step": step, "eval": evaluate_info}, default=lambda value: '
        'float(value.item()) if hasattr(value, "item") else str(value)), flush=True)\n'
        + anchor
    )
    if source.count(anchor) != 1:
        raise RuntimeError(f"Evaluation logging anchor changed in {known.trainer}")
    source = source.replace(anchor, replacement)

    sys.argv = [str(known.trainer), *trainer_args]
    namespace = {
        "__name__": "__main__",
        "__file__": str(known.trainer),
        "__package__": None,
        "__cached__": None,
    }
    os.chdir(known.trainer.parent)
    # The source is a user-selected local trainer, not untrusted input.
    exec(compile(source, str(known.trainer), "exec"), namespace)  # noqa: S102


if __name__ == "__main__":
    main()
