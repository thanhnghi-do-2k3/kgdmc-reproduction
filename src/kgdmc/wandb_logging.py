from __future__ import annotations

import os
from typing import Any

import wandb


def start_run(name: str, config: dict[str, Any]) -> Any:
    # Local tests and cloud jobs must never prompt for credentials implicitly.
    # Opt into online tracking explicitly with WANDB_MODE=online after `wandb login`.
    if os.getenv("WANDB_MODE", "disabled") == "disabled":
        return None
    return wandb.init(
        project=os.getenv("WANDB_PROJECT", "graph-paper-2-kgdmc"),
        entity=os.getenv("WANDB_ENTITY") or None,
        name=name,
        group=os.getenv("WANDB_RUN_GROUP") or None,
        job_type=os.getenv("WANDB_JOB_TYPE", "reproduction"),
        mode=os.getenv("WANDB_MODE", "disabled"),
        config=config,
        reinit="return_previous",
    )


def log(run: Any, values: dict[str, Any], step: int | None = None) -> None:
    if run is not None:
        run.log(values, step=step)


def finish(run: Any, summary: dict[str, Any] | None = None) -> None:
    if run is not None:
        if summary:
            run.summary.update(summary)
        run.finish()
