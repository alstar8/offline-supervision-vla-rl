import os
import time
from typing import Any, Dict

import wandb


def init_wandb_with_online_fallback(
    config: Dict[str, Any],
    project: str,
    name: str,
    use_wandb: bool,
    max_online_attempts: int = 2,
    retry_sleep_sec: float = 3.0,
) -> None:
    if not use_wandb:
        os.environ["WANDB_MODE"] = "offline"
        wandb.init(config=config, project=project, name=name, mode="offline")
        return

    attempts = max(1, int(max_online_attempts))
    for attempt in range(1, attempts + 1):
        try:
            os.environ.pop("WANDB_MODE", None)
            wandb.init(config=config, project=project, name=name, mode="online")
            return
        except Exception as exc:
            print(
                f"[W&B] Online init attempt {attempt}/{attempts} failed: {type(exc).__name__}: {exc}"
            )
            if wandb.run is not None:
                try:
                    wandb.finish()
                except Exception:
                    pass
            if attempt < attempts:
                time.sleep(retry_sleep_sec)
                continue

            os.environ["WANDB_MODE"] = "offline"
            print("")
            print("=" * 100)
            print("WARNING: WANDB ONLINE CONNECTION FAILED TWICE. FALLING BACK TO OFFLINE MODE.")
            print("WARNING: THIS RUN WILL NOT LIVE-STREAM METRICS TO WANDB UNTIL MANUALLY SYNCED.")
            print("=" * 100)
            print("")
            wandb.init(config=config, project=project, name=name, mode="offline")
            return
