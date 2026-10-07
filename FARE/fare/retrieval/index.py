from __future__ import annotations

from pathlib import Path
import json
import gc
import numpy as np
import torch

from ..utils import write_json


def _cuda_release():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


def _initial_micro_batch(device: torch.device, requested: int) -> int:
    requested = max(1, int(requested))
    if device.type != "cuda" or not torch.cuda.is_available():
        return requested
    try:
        free_b, total_b = torch.cuda.mem_get_info(device)
        free_gb = float(free_b) / (1024 ** 3)
        if free_gb < 4.0:
            cap = 2
        elif free_gb < 6.0:
            cap = 4
        elif free_gb < 10.0:
            cap = 8
        elif free_gb < 14.0:
            cap = 16
        elif free_gb < 22.0:
            cap = 32
        else:
            cap = 64
        chosen = max(1, min(requested, cap))
        print(
            f"RETRIEVAL_MEMORY_PROFILE device={torch.cuda.get_device_name(device)} "
            f"free_gb={free_gb:.2f} total_gb={float(total_b)/(1024**3):.2f} "
            f"requested_batch={requested} initial_micro_batch={chosen}",
            flush=True,
        )
        return chosen
    except Exception:
        return min(requested, 8)


def _is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


class LearnedRetrievalIndex:
    """Learned memory index backed by ContextRestorer global embeddings."""
    def __init__(self, ctx, index_dir: str | Path):
        self.ctx = ctx
        self.index_dir = Path(index_dir)
        meta = json.loads((self.index_dir / "index_meta.json").read_text(encoding="utf-8"))
        if meta["backbone_checkpoint_sha256"] != ctx.checkpoint_sha256:
            raise RuntimeError("Retrieval index Backbone checkpoint fingerprint mismatch")
        self.starts = np.load(self.index_dir / "starts.npy", mmap_mode="r").astype(np.int64)
        self.embeddings_np = np.load(self.index_dir / "embeddings.npy", mmap_mode="r")
        self.memory_values = np.load(self.index_dir / "memory_values_f16.npy", mmap_mode="r")
        self.memory_valid = np.load(self.index_dir / "memory_valid_u8.npy", mmap_mode="r")
        self._embeddings_device = None
        self._embeddings = None
        self.target_idx = int(ctx.bench.target_idx)
        self.valid = ctx.bench.valid
        self.traffic = ctx.bench.traffic

    @classmethod
    def build(cls, ctx, context_model, out_dir: str | Path, memory_rows: int, batch_size: int=128):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        H = int(ctx.cfg["experiment"]["history"])
        train_range = ctx.bench.split_range("train")
        train_start = int(train_range["start"]); train_end = int(train_range["end_exclusive"])
        starts = np.arange(train_start, train_end - H + 1, dtype=np.int64)
        target_idx = int(ctx.bench.target_idx)
        from experiment_common.time_features import resolve_time_features
        tf_cfg = ctx.backbone_cfg.get('time_features', {'allow_index_modulo_fallback': True, 'index_origin_dow': 0})
        tf = resolve_time_features(ctx.bench, allow_index_fallback=bool(tf_cfg.get('allow_index_modulo_fallback', True)), index_origin_dow=int(tf_cfg.get('index_origin_dow', 0)))
        valid_target = np.asarray(ctx.bench.valid[..., target_idx], dtype=bool)
        step_valid = valid_target.sum(axis=1, dtype=np.int64)
        prefix = np.concatenate([[0], np.cumsum(step_valid, dtype=np.int64)])
        counts = prefix[starts + H] - prefix[starts]
        coverage = (counts.astype(np.float64) / float(H * valid_target.shape[1])).astype(np.float32)
        good = starts[coverage >= 0.80]
        if len(good) == 0:
            raise RuntimeError("No train windows satisfy retrieval memory coverage >= 0.80")
        if len(good) > int(memory_rows):
            pos = np.linspace(0, len(good)-1, int(memory_rows)).round().astype(np.int64)
            good = good[pos]
        starts = np.asarray(good, np.int64)
        device = ctx.device
        context_model.eval()
        _cuda_release()
        micro_bs = _initial_micro_batch(device, int(batch_size))

        mem_values = np.lib.format.open_memmap(out / 'memory_values_f16.npy', mode='w+', dtype=np.float16, shape=(len(starts), H, int(ctx.bench.N)))
        mem_valid = np.lib.format.open_memmap(out / 'memory_valid_u8.npy', mode='w+', dtype=np.uint8, shape=(len(starts), H, int(ctx.bench.N)))
        emb_map = np.lib.format.open_memmap(out / 'embeddings.npy', mode='w+', dtype=np.float32, shape=(len(starts), int(context_model.encoder.hidden_dim)))

        done = 0
        with torch.inference_mode():
            while done < len(starts):
                cur = min(micro_bs, len(starts) - done)
                ss = starts[done:done+cur]
                try:
                    xb=[]; mb=[]; tod=[]; dow=[]
                    for st in ss:
                        raw=np.asarray(ctx.bench.traffic[st:st+H,:,target_idx],np.float32)
                        vm=np.asarray(ctx.bench.valid[st:st+H,:,target_idx],bool)
                        xn=(raw-ctx.mean)/ctx.std
                        xb.append(np.where(vm,xn,0.0).astype(np.float32)); mb.append(vm)
                        tod.append(np.asarray(tf.tod[int(st):int(st)+H], np.int64)); dow.append(np.asarray(tf.dow[int(st):int(st)+H], np.int64))
                    xb_np=np.stack(xb); mb_np=np.stack(mb)
                    mem_values[done:done+cur] = xb_np.astype(np.float16)
                    mem_valid[done:done+cur] = mb_np.astype(np.uint8)
                    x=torch.from_numpy(xb_np).to(device, non_blocking=True)
                    m=torch.from_numpy(mb_np).to(device, non_blocking=True)
                    td=torch.from_numpy(np.stack(tod)).to(device, non_blocking=True)
                    dw=torch.from_numpy(np.stack(dow)).to(device, non_blocking=True)
                    z=context_model.global_embedding(x,m,td,dw)
                    emb_map[done:done+cur] = z.detach().cpu().numpy().astype(np.float32)
                    del x,m,td,dw,z,xb,mb,tod,dow,xb_np,mb_np
                    done += cur
                    if done % max(500, micro_bs*20) < cur or done >= len(starts):
                        print(f"RETRIEVAL_EMBED {done}/{len(starts)} micro_batch={micro_bs}", flush=True)
                except RuntimeError as exc:
                    if device.type == 'cuda' and _is_cuda_oom(exc) and micro_bs > 1:
                        new_bs=max(1,micro_bs//2)
                        print(f"RETRIEVAL_EMBED_OOM batch={micro_bs} -> retry_batch={new_bs}", flush=True)
                        # Drop references from the failed chunk before releasing the CUDA cache.
                        for _name in ('x','m','td','dw','z'):
                            if _name in locals():
                                locals()[_name] = None
                        micro_bs=new_bs
                        _cuda_release()
                        continue
                    raise

        mem_values.flush(); mem_valid.flush(); emb_map.flush()
        np.save(out/"starts.npy",starts)
        write_json(out/"index_meta.json",{
            "index_version":"fare_v023_context_embedding_v1",
            "runtime_patch":"0.2.3_memory_safe_embedding",
            "rows":int(len(starts)),
            "embedding_dim":int(emb_map.shape[1]),
            "backbone_checkpoint_sha256":ctx.checkpoint_sha256,
            "context_checkpoint":str(ctx.artifact_root/"context"/"best.pt"),
            "support_policy":"dense_stride1_chronological_train_windows_only",
            "min_history_coverage":0.80,
            "final_embedding_micro_batch":int(micro_bs),
        })
        return out

    def _emb(self, device):
        if self._embeddings is None or self._embeddings_device != str(device):
            self._embeddings=torch.from_numpy(np.asarray(self.embeddings_np,np.float32).copy()).to(device)
            self._embeddings_device=str(device)
        return self._embeddings

    def search(self, query_embedding: torch.Tensor, query_starts: torch.Tensor, top_k: int, search_k: int):
        mem=self._emb(query_embedding.device)
        score=query_embedding @ mem.T
        starts=torch.from_numpy(np.asarray(self.starts,np.int64).copy()).to(query_embedding.device)
        H=int(self.ctx.cfg["experiment"]["history"])
        causal=(starts[None,:] + H) <= query_starts[:,None]
        score=score.masked_fill(~causal,-1e9)
        k=min(int(search_k),score.shape[1])
        vals,inds=torch.topk(score,k=k,dim=1)
        if torch.any(vals[:,:min(int(top_k),k)] < -1e8):
            raise RuntimeError("Insufficient causal retrieval candidates for at least one query")
        return inds[:,:int(top_k)], vals[:,:int(top_k)]

    def materialize(self, indices: torch.Tensor):
        inds=indices.detach().cpu().numpy().astype(np.int64)
        x=np.asarray(self.memory_values[inds],dtype=np.float32)
        v=np.asarray(self.memory_valid[inds],dtype=np.uint8).astype(bool)
        return x,v
