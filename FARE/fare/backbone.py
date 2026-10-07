from __future__ import annotations

import numpy as np
import torch


class FrozenBackbone:
    def __init__(self, adapter, batch_size: int = 64):
        self.adapter = adapter
        self.batch_size = max(1, int(batch_size))
        self.device = adapter.device

    def forward_norm_tensor(self, x: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        batch = {"x": x, "tod_hist": tod, "dow_hist": dow}
        return self.adapter.eval_forward(batch)

    @torch.no_grad()
    def predict_norm_tensor(self, x: torch.Tensor, tod: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        return self.forward_norm_tensor(x, tod, dow)

    @torch.inference_mode()
    def predict_raw(self, x, tod_hist, dow_hist):
        x = np.asarray(x, np.float32); tod_hist=np.asarray(tod_hist,np.int64); dow_hist=np.asarray(dow_hist,np.int64)
        out=[]
        for s in range(0,len(x),self.batch_size):
            xb=torch.from_numpy(np.asarray(x[s:s+self.batch_size],np.float32).copy()).to(self.device)
            tb=torch.from_numpy(np.asarray(tod_hist[s:s+self.batch_size],np.int64).copy()).to(self.device)
            db=torch.from_numpy(np.asarray(dow_hist[s:s+self.batch_size],np.int64).copy()).to(self.device)
            pn=self.forward_norm_tensor(xb,tb,db)
            out.append(self.adapter.raw(pn).detach().cpu().numpy().astype(np.float32))
        return np.concatenate(out,0)
