"""Safe evaluation runtime for existing PCFT v1.0 checkpoints.

Resource settings are NOT model architecture settings. These overrides never
alter the trained parameters, saved checkpoint, prediction scaling, or retrieval
candidate policy; they only change how many queries are evaluated at once.
"""
from __future__ import annotations
import copy
import gc
import torch


def safe_evaluation_config(saved_cfg, batch_size=None, num_workers=0,
                           preload_repository_gpu=False):
    cfg = copy.deepcopy(saved_cfg)
    rt = cfg.setdefault("runtime", {})
    tr = cfg.setdefault("training", {})
    requested = batch_size if batch_size is not None else rt.get("eval_batch_size", 8)
    bs = int(requested)
    if bs < 1:
        raise ValueError("Evaluation batch size must be >= 1")
    nw = int(num_workers)
    if nw < 0:
        raise ValueError("Evaluation workers cannot be negative")
    tr["batch_size"] = bs
    rt["num_workers"] = nw
    rt["persistent_workers"] = False
    rt["prefetch_factor"] = 2
    rt["pin_memory"] = bool(nw > 0 and torch.cuda.is_available())
    rt["preload_repository_gpu"] = bool(preload_repository_gpu)
    return cfg


def is_cuda_oom(error):
    return isinstance(error, RuntimeError) and (
        isinstance(error, torch.cuda.OutOfMemoryError)
        or "cuda out of memory" in str(error).lower()
        or "cublas_status_alloc_failed" in str(error).lower()
    )


def predict_with_retry(model, dataset, index, repo, device, mean, std, mode,
                       amp, cfg, make_loader, predict, start_batch_size):
    """Retry an entire severity deterministically after CUDA OOM.

    Prior partially collected predictions are discarded on every failed pass.
    This can never silently omit queries or change the model weights.
    """
    bs = int(start_batch_size)
    while True:
        loader = make_loader(dataset, bs, False, cfg)
        try:
            result = predict(model, loader, index, repo, device,
                             mean, std, mode, amp=amp)
            if len(result["prediction"]) != len(dataset):
                raise RuntimeError("Incomplete evaluation: prediction count mismatches dataset")
            return result, bs
        except RuntimeError as exc:
            if not is_cuda_oom(exc) or bs == 1:
                raise
            next_bs = max(1, bs // 2)
            print(f"EVAL_OOM_RETRY severity_batches={len(dataset)} batch={bs} -> {next_bs} "
                  f"reason={str(exc).splitlines()[0]}", flush=True)
            bs = next_bs
        finally:
            del loader
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
