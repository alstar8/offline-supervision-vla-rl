import argparse
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq

from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead


def _dtype_from_str(name: str) -> torch.dtype:
    name = name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name in ("float16", "fp16"):
        return torch.float16
    if name in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unknown torch_dtype: {name}")


def _load_model(path: str, model_class: str, vh_mode: str, torch_dtype: torch.dtype, device_map: str):
    if model_class == "action_value":
        return OpenVLAForActionPredictionWithValueHead.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
            vh_mode=vh_mode,
        )
    if model_class == "vision2seq":
        return AutoModelForVision2Seq.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=device_map,
        )
    raise ValueError(f"Unknown model_class: {model_class}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print model parameter names and shapes.")
    parser.add_argument("--model_path", required=True, help="HF hub id or local path")
    parser.add_argument("--model_class", default="vision2seq", choices=["vision2seq", "action_value"])
    parser.add_argument("--vh_mode", default="a0")
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", default="cpu")
    parser.add_argument("--output_path", default="", help="Optional file to write output")
    args = parser.parse_args()

    torch_dtype = _dtype_from_str(args.torch_dtype)
    model = _load_model(args.model_path, args.model_class, args.vh_mode, torch_dtype, args.device_map)

    lines = []
    for name, param in model.named_parameters():
        lines.append(f"{name}\t{tuple(param.shape)}\t{param.dtype}")

    out_text = "\n".join(lines)
    if args.output_path:
        Path(args.output_path).write_text(out_text)
        print(f"Wrote {len(lines)} params to {args.output_path}")
    else:
        print(out_text)


if __name__ == "__main__":
    main()
