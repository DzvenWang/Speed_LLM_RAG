import torch
from online_presure.updater import configure_trainable

class M(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.spa_encoder=torch.nn.ModuleList([torch.nn.Linear(4,4),torch.nn.Linear(4,4)]); self.regression_conv=torch.nn.Conv2d(4,1,1)

m=M(); p=configure_trainable(m,'last_block_head'); n=sum(x.numel() for x in p); assert n>0 and all(x.requires_grad for x in m.spa_encoder[-1].parameters()) and all(not x.requires_grad for x in m.spa_encoder[0].parameters())
print('UPDATER_SMOKE PASS trainable=',n)
