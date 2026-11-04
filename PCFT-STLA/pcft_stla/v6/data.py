from __future__ import annotations
import json, os, hashlib, shutil
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import Dataset
from .config import sha256,json_write


def open_bench(repo, dataset, target=None, benchmark_root=None):
    """Open the target-specific SupportBench base dataset without rebuilding it.

    The Base_SupportBench/scripts/build_dataset.py launcher writes to:
      data/processed/traffident/base_supportbench/<target>/<dataset>/
    where <target> is speed or flow.  The upstream *default YAML* still
    contains the older target-agnostic root, so using its value verbatim can
    point to a non-existent directory on Gadi.

    PCFT_BENCHMARK_ROOT may override the benchmark parent. It accepts either
    the shared `base_supportbench` directory or its `<target>` child.  An
    explicit override is authoritative: a typo fails rather than silently
    training on a different dataset.  `benchmark_root` takes precedence over
    the environment variable.
    """
    import sys
    repo = Path(repo).expanduser().resolve()
    package_root = repo / 'Base_SupportBench'
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from supportbench.dataset import SupportBench

    target = str(target or os.environ.get('TRAFFIDENT_TARGET', '')).strip().lower()
    if target not in ('speed', 'flow'):
        raise ValueError('open_bench requires target=speed|flow; refusing to guess the target-specific dataset')

    override = benchmark_root or os.environ.get('PCFT_BENCHMARK_ROOT', '').strip()
    candidates = []
    if override:
        specified = Path(os.path.expandvars(os.path.expanduser(str(override))))
        if not specified.is_absolute():
            specified = repo / specified
        specified = specified.resolve()
        # The caller may specify the dataset itself, the target root, or the
        # shared parent. All three are unambiguous once target is known.
        if specified.name == dataset:
            candidates.append(specified.parent)
        elif specified.name == target:
            candidates.append(specified)
        else:
            candidates.append(specified / target)
            candidates.append(specified)
    else:
        # The canonical 2026 Base_SupportBench build_dataset.py location.
        candidates.append(repo / 'data/processed/traffident/base_supportbench' / target)
        # Prefer the path persisted by the target's resolved build config if
        # the benchmark was moved after preprocessing.
        resolved = (repo / 'Base_SupportBench_artifacts/resolved_configs'
                    / target / dataset / 'base_supportbench.yaml')
        if resolved.is_file():
            from supportbench.config import load_config
            resolved_cfg = load_config(resolved)
            configured = Path(resolved_cfg['project']['benchmark_root']).expanduser().resolve()
            candidates.insert(0, configured)
        # Support older single-target builds only when their manifest actually
        # declares the requested target, never cross-use a speed source for flow.
        candidates.append(repo / 'data/processed/traffident/base_supportbench')

    seen = set()
    checked = []
    for raw_root in candidates:
        base = Path(raw_root).resolve()
        if base in seen:
            continue
        seen.add(base)
        sg = base / dataset
        checked.append(str(sg))
        manifest = sg / 'manifest.json'
        if not manifest.is_file():
            continue
        j = json.loads(manifest.read_text(encoding='utf-8'))
        actual = str(j.get('primary_channel', '')).lower()
        if actual != target:
            raise RuntimeError(
                f'SupportBench target mismatch at {sg}: requested={target}, '
                f'manifest.primary_channel={actual}. Refusing cross-target sensing masks.'
            )
        bench = SupportBench(dataset, root=base)
        if target not in bench.channel_names:
            raise ValueError(f'SupportBench {sg} has no {target} channel: {bench.channel_names}')
        print(f'PCFT_BENCHMARK_ROOT_RESOLVED target={target} path={bench.sg_dir}', flush=True)
        return bench

    raise FileNotFoundError(
        f'No completed {target} SupportBench dataset for {dataset}. Checked:\n  '
        + '\n  '.join(checked)
        + '\nExpected a manifest.json plus core/, graph/, views/ in the '
          'target-specific dataset. Check `ls -ld '
          f'data/processed/traffident/base_supportbench/{target}/{dataset}`; '
          'if the dataset exists elsewhere, set PCFT_BENCHMARK_ROOT '
          'to the parent or target directory. If it has not been built, '
          'restore/rebuild the canonical Base_SupportBench data; '
          'an existing ExperimentCache alone is not sufficient for v5.'
    )


def make_multi_cache(bench,target_root,split,out,*,max_rows=None):
    """One-time multi-channel materialization aligned exactly by immutable view_id.

    This is an INPUT cache only: no valid future traffic is stored in x/obs.
    """
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(Path(target_root)/f'{split}_manifest.csv',dtype={'view_id':str})
    n=len(meta) if max_rows is None else min(len(meta),int(max_rows))
    meta=meta.iloc[:n].reset_index(drop=True)
    if 'view_id' not in meta or meta.view_id.isnull().any() or meta.view_id.duplicated().any():
        raise ValueError(f'{split} manifest requires unique non-null view_id')
    source_ids=set(bench.views.view_id.astype(str))
    if any(str(x) not in source_ids for x in meta.view_id):raise ValueError(f'{split}: target cache view_id not present in benchmark')
    fingerprint=hashlib.sha256(('|'.join(meta.view_id.astype(str))+'|'+str(bench.sg_dir)+'|'+str(bench.manifest.get('version',''))+'|'+str(bench.channel_names)).encode()).hexdigest()
    manifest=out/'identity.json'
    if manifest.exists():
        j=json.loads(manifest.read_text())
        if j['fingerprint']==fingerprint and j['n']==n and j['channels']==bench.channel_names:
            files=[out/'x.npy',out/'obs.npy',out/'age.npy']
            if all(p.exists() for p in files) and np.load(files[0],mmap_mode='r').shape==(n,bench.H,bench.N,len(bench.channel_names)):
                return out
        raise RuntimeError(f'Existing multivariate cache has different identity; manually delete only {out}')
    h=bench.H; nodes=bench.N;c=len(bench.channel_names)
    if out.exists():
        raise RuntimeError(f'Unrecognized or incomplete multi-cache at {out}; inspect before removing')
    stage=out.parent/(out.name+'.building')
    if stage.exists():shutil.rmtree(stage)
    stage.mkdir(parents=True)
    x=np.lib.format.open_memmap(stage/'x.npy',mode='w+',dtype=np.float32,shape=(n,h,nodes,c))
    obs=np.lib.format.open_memmap(stage/'obs.npy',mode='w+',dtype=np.uint8,shape=(n,h,nodes,c))
    ages=np.lib.format.open_memmap(stage/'age.npy',mode='w+',dtype=np.uint8,shape=(n,h,nodes,c))
    x_target=np.load(Path(target_root)/f'{split}_x_obs_norm.npy',mmap_mode='r')
    m_target=np.load(Path(target_root)/f'{split}_history_observed.npy',mmap_mode='r')
    # Use the TARGET cache channel for mask materialization, not benchmark primary_channel.
    cache_manifest=json.loads((Path(target_root)/'cache_manifest.json').read_text())
    name=str(cache_manifest.get('target_name',bench.target_name))
    target_idx=bench.channel_names.index(name)
    st=bench.stats['channels'][name]
    try:
        for i,row in enumerate(meta.itertuples(index=False)):
            spec=bench.get_view_spec(str(row.view_id))
            window=int(spec['window_start'])
            if window!=int(row.window_start) or str(spec['split'])!=split:
                raise RuntimeError(f'{split} row {i}: source and target window mismatch')
            hist=np.asarray(bench.traffic[window:window+bench.H],dtype=np.float32)
            raw_valid=np.asarray(bench.valid[window:window+bench.H],dtype=bool)
            from supportbench.sensing import materialize_mask
            art=materialize_mask(spec,bench.H,bench.N,bench.sg_dir,raw_valid[...,target_idx])
            mi=raw_valid & (~art[...,None])
            xi=bench.normalize(hist)
            if not np.isfinite(xi[mi]).all():
                raise RuntimeError(f'{split} row {i}: nonfinite valid multivariate observations')
            xi=np.where(mi,xi,0.).astype(np.float32)
            # Independently audit SAME target channel history and validity as the official shared cache.
            expect=np.asarray(x_target[i],np.float32)
            check=(hist[...,target_idx]-float(st['mean']))/max(float(st['std']),1e-6)
            check=np.where(mi[...,target_idx],check,0.)
            if not np.allclose(check,expect,atol=1e-4,rtol=1e-4) or not np.array_equal(mi[...,target_idx],np.asarray(m_target[i],bool)):
                raise RuntimeError(f'INPUT ALIGNMENT FAILED at {split}/{i}/{name}: normalized history or mask differs')
            if not np.all((~mi)|raw_valid):raise RuntimeError('Observed invalid source values')
            x[i]=np.where(mi,xi,0);obs[i]=mi.astype(np.uint8)
            ages[i]=np.rint(age_sequence(mi)*h).astype(np.uint8)
            if (i+1)%1000==0:print(f'V6_CACHE {split} {i+1}/{n}',flush=True)
        x.flush();obs.flush();ages.flush()
        json_write(stage/'identity.json',{'fingerprint':fingerprint,'n':n,'channels':bench.channel_names,'split':split,'target_name':name,'target_manifest_sha256':sha256(Path(target_root)/f'{split}_manifest.csv'),'benchmark_root':str(bench.sg_dir)})
        stage.replace(out)
    except BaseException:
        del x,obs,ages
        shutil.rmtree(stage,ignore_errors=True)
        raise
    return out


def age_sequence(mask):
    # [H,N,C]; capped to H, 1 denotes no recent observed value.
    H=mask.shape[0]
    age=np.zeros(mask.shape,np.float32)
    prev=np.zeros(mask.shape[1:],np.float32)
    for t in range(H):
        prev=np.where(mask[t],0,np.minimum(prev+1,H))
        age[t]=prev/float(H)
    return age


class MultiTargetSplit(Dataset):
    def __init__(self,target_root,split,multi_root,*,limit=None):
        root=Path(target_root);self.root=root;self.split=split
        self.meta=pd.read_csv(root/f'{split}_manifest.csv',dtype={'view_id':str})
        self.n=len(self.meta) if limit is None else min(len(self.meta),int(limit))
        self.x=np.load(Path(multi_root)/'x.npy',mmap_mode='r');self.obs=np.load(Path(multi_root)/'obs.npy',mmap_mode='r')
        self.ages=np.load(Path(multi_root)/'age.npy',mmap_mode='r')
        if len(self.x)!=self.n or len(self.obs)!=self.n or len(self.ages)!=self.n:raise ValueError('Multi cache rows do not match target cache limit')
        self.y=np.load(root/f'{split}_y_norm.npy',mmap_mode='r')
        self.raw=np.load(root/f'{split}_y_raw.npy',mmap_mode='r')
        self.valid=np.load(root/f'{split}_future_valid.npy',mmap_mode='r')
        self.direct=np.load(root/f'{split}_direct_mask.npy',mmap_mode='r')
        self.tod=np.load(root/f'{split}_tod_hist.npy',mmap_mode='r');self.dow=np.load(root/f'{split}_dow_hist.npy',mmap_mode='r')
        self.todf=np.load(root/f'{split}_tod_future.npy',mmap_mode='r');self.dowf=np.load(root/f'{split}_dow_future.npy',mmap_mode='r')
        ident=json.loads((Path(multi_root)/'identity.json').read_text())
        if ident['target_manifest_sha256']!=sha256(root/f'{split}_manifest.csv'):raise RuntimeError('Data cache manifest fingerprint mismatch')
        self.channels=list(ident['channels'])
        self.target=str(ident['target_name'])
        self.target_idx=self.channels.index(self.target)
    def __len__(self):return self.n
    def __getitem__(self,i):
        x=np.asarray(self.x[i],np.float32).copy();obs=np.asarray(self.obs[i],np.uint8).astype(bool)
        x=np.where(obs,x,0).astype(np.float32)
        return dict(
            x_multi=torch.from_numpy(x),observed_mask=torch.from_numpy(obs.astype(np.float32)),
            missing_age=torch.from_numpy(np.asarray(self.ages[i],np.float32).copy()/float(x.shape[0])),
            x_target=torch.from_numpy(x[...,self.target_idx].copy()),
            y_norm=torch.from_numpy(np.asarray(self.y[i],np.float32).copy()),
            y_raw=torch.from_numpy(np.asarray(self.raw[i],np.float32).copy()),
            future_valid=torch.from_numpy(np.asarray(self.valid[i],bool).copy()),
            direct_mask=torch.from_numpy(np.asarray(self.direct[i],bool).copy()),
            tod=torch.from_numpy(np.asarray(self.tod[i],np.int64).copy()),
            dow=torch.from_numpy(np.asarray(self.dow[i],np.int64).copy()),
            tod_future=torch.from_numpy(np.asarray(self.todf[i],np.int64).copy()),
            dow_future=torch.from_numpy(np.asarray(self.dowf[i],np.int64).copy()),
            index=torch.tensor(i,dtype=torch.long))
