"""torchrun entry: pin one physical GPU per rank, then start RefKL (SFT trainer)."""
import os

_gpus = [g.strip() for g in os.environ.get("REFKL_GPUS", "0,1,2,4,5,6,7").split(",") if g.strip()]
_local_rank = int(os.environ.get("LOCAL_RANK", "0"))
if _local_rank < 0 or _local_rank >= len(_gpus):
    raise RuntimeError(f"LOCAL_RANK={_local_rank} out of range for REFKL_GPUS={_gpus}")
os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[_local_rank]

from simpler_env.train_ms3_ppo_sft import main

if __name__ == "__main__":
    main()
