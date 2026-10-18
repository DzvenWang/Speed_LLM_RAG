"""Run one FARE model through memory, staged training, automatic evaluation and internal attribution."""
from pathlib import Path
import time,os,json,torch
from .config import root,cache_dir,backbone_checkpoint,hash_file,dump_json,project_path
from .protocol import SupportDataset,graph_files,directed_metadata
from .traffic import TrafficGraph
from .backbone import load_stla
from .memory import EpisodeMemory,from_support_cache
from .model import FARE
from .cuda_preflight import run_cuda_preflight
from .training import fit,seed_everything
from .evaluation import evaluate
from .analysis import analyze

def prepare(cfg,device,backbone_file=None,memory_file=None,build=True,oos_forecast_file=None,cuda_safety=False):
    ex=cfg['experiment'];base_cache=cache_dir(cfg,'formal')
    train=SupportDataset(base_cache,'train',ex.get('train_limit'),strategy=ex.get('sample_strategy','spread'))
    val=SupportDataset(base_cache,'val',ex.get('val_limit'),strategy='spread')
    adjacency=graph_files(base_cache)['adj']
    if adjacency.shape[0]!=train.n or val.n!=train.n:raise RuntimeError('Graph / dataset node count mismatch')
    direct=directed_metadata(ex['dataset'],ex['target'],root())
    graph=TrafficGraph(adjacency,direct,cfg['traffic']['region_size'],
                       cfg['traffic']['neighbor_count'],cfg['traffic'].get('max_hops',3),
                       cfg['traffic'].get('path_sources',16)).to(device)
    graph_index=root()/'FARE_cache'/ex['dataset']/'shared_graph'/'sparse_paths.npz'
    if not graph_index.exists():graph.save_index(graph_index)
    bk=backbone_checkpoint(cfg,backbone_file)
    backbone=load_stla(bk,device,cfg['backbone']['inject_layers'])
    if backbone.n!=train.n:raise RuntimeError('Checkpoint and dataset node count mismatch')
    if cuda_safety and cfg.get('training',{}).get('cuda_preflight',True):
        run_cuda_preflight(backbone,graph,train,cfg,device)
    mem_path=project_path(memory_file) if memory_file else root()/'FARE_cache'/ex['target']/ex['dataset']/f"memory_v4_{ex['seed']}_{cfg['memory']['max_episodes']}"
    expected_memory={'dataset':ex['dataset'],'target':ex['target'],'backbone_sha256':hash_file(bk)}
    if oos_forecast_file and cfg['memory']['residual_mode']=='external_oos':
        expected_memory['oos_source_sha256']=hash_file(oos_forecast_file)
    if mem_path.exists() and (mem_path/'manifest.json').exists():
        memory=EpisodeMemory.load(mem_path,expected=expected_memory)
    elif build:
        memory=from_support_cache(base_cache,backbone,cfg['memory']['max_episodes'],
                                  cfg['memory']['residual_mode'],oos_forecast_file,device,
                                  cfg['memory']['build_batch_size'])
        memory.metadata.update({'dataset':ex['dataset'],'target':ex['target'],'backbone_sha256':hash_file(bk)})
        if oos_forecast_file and cfg['memory']['residual_mode']=='external_oos':
            memory.metadata['oos_source_sha256']=hash_file(oos_forecast_file)
        mem_path.parent.mkdir(parents=True,exist_ok=True);memory.save(mem_path)
    else:raise FileNotFoundError(mem_path)
    memory.attach_graph(graph)
    if cfg['experiment']['run_tag']=='formal' and cfg['memory']['require_oos_for_formal'] and memory.metadata.get('residual_provenance')!='external_oos':
        raise RuntimeError('Formal FARE requires genuinely held-out/OOF residuals. Supply external_oos source.')
    model=FARE(backbone,graph,cfg).to(device)
    metadata={'dataset':ex['dataset'],'target':ex['target'],'seed':ex['seed'],
              'backbone_checkpoint':str(bk),'backbone_sha256':hash_file(bk),
              'memory_path':str(mem_path),'memory_sha256':hash_file(mem_path/'manifest.json'),
              'residual_provenance':memory.metadata.get('residual_provenance'),
              'direction_provenance':graph.metadata_source,'nodes':train.n}
    return model,memory,train,val,device,metadata

def create_run_dir(cfg,explicit=None):
    if explicit:return project_path(explicit)
    ex=cfg['experiment'];stamp=time.strftime('%Y%m%d_%H%M%S')
    return root()/'FARE_exps'/ex['target']/ex['dataset']/ex['run_tag']/f'{stamp}_seed{ex["seed"]}'

def evaluations(cfg,limit_override=None):
    sets={}
    for severity in cfg['experiment']['severities']:
        path=cache_dir(cfg,'evaluation',severity)
        sets[f's{severity:02}']=SupportDataset(path,'test',limit_override if limit_override is not None else cfg['experiment'].get('test_limit'),strategy='spread')
    return sets

def post_training_datasets(cfg,val_ds=None,limit_override=None):
    """Only selected post-training evaluation/attribution splits.

    Training's Validation loss/checkpoint selection is separate and always retained.
    """
    test_sets=evaluations(cfg,limit_override)
    if cfg.get('evaluation',{}).get('include_val',False):
        if val_ds is None:raise ValueError('val_ds required when evaluation.include_val=true')
        return {'val':val_ds,**test_sets}
    return test_sets

def run(cfg,device,run_dir=None,backbone_file=None,memory_file=None,oos_forecast_file=None,resume=None,
        train=True,eval_only=False,analysis=True):
    seed_everything(cfg['experiment']['seed']);run_dir=create_run_dir(cfg,run_dir)
    if resume is not None:resume=project_path(resume,must_exist=True)
    run_dir.mkdir(parents=True,exist_ok=True)
    model,memory,train_ds,val_ds,device,meta=prepare(cfg,device,backbone_file,memory_file,True,project_path(oos_forecast_file,must_exist=True) if oos_forecast_file else None,
                                                         cuda_safety=train)
    dump_json(run_dir/'manifest.json',{'config':cfg,'metadata':meta,'run_dir':str(run_dir),'version':'4.0.4'})
    if train:
        best=fit(model,memory,train_ds,val_ds,device,cfg,run_dir,meta,resume)
    else:best=project_path(resume,must_exist=True) if resume else run_dir/'best.pt'
    ck=torch.load(best,map_location=device,weights_only=False)
    if ck['metadata']['backbone_sha256']!=meta['backbone_sha256'] or ck['metadata']['memory_sha256']!=meta['memory_sha256']:
        raise RuntimeError('Backbone / Memory provenance differs from best checkpoint')
    model.load_state_dict(ck['state_dict'],strict=True)
    if cfg['evaluation']['auto_evaluate'] or eval_only:
        datasets=post_training_datasets(cfg,val_ds)
        print('FARE_POST_TRAINING_SPLITS '+json.dumps(list(datasets)),flush=True)
        performance=[]
        for name,ds in datasets.items():
            res,_=evaluate(model,memory,ds,device,cfg,run_dir/'evaluation'/name,save_traces=True)
            performance.append({'split':name,'MAE':res.get('overall_MAE'),'RMSE':res.get('overall_RMSE'),
                                'MAPE':res.get('overall_MAPE'),'directly_masked_MAE':res.get('masked_MAE'),
                                'directly_masked_RMSE':res.get('masked_RMSE')})
            print('FARE_EVALUATION '+json.dumps({'split':name,'overall_MAE':res.get('overall_MAE'),
                                                   'overall_RMSE':res.get('overall_RMSE')}),flush=True)
        from .analysis import write_csv
        test_rows=[r for r in performance if r['split'].startswith('s')]
        for name,subset in [('Avg_s20_s90',test_rows),('Avg_s20_s60',[r for r in test_rows if r['split']!='s90'])]:
            if len(subset)==(4 if name=='Avg_s20_s90' else 3):
                performance.append({'split':name,**{k:sum(float(r[k]) for r in subset)/len(subset)
                                                    if all(r[k] is not None for r in subset) else None
                                                    for k in ('MAE','RMSE','MAPE','directly_masked_MAE','directly_masked_RMSE')}})
        write_csv(run_dir/'evaluation'/'performance_summary.csv',performance)
        print('FARE_EVALUATE_COMPLETE',flush=True)
        if analysis and cfg['analysis']['enabled']:
            analyze(model,memory,datasets,device,cfg,run_dir/'analysis')
    dump_json(run_dir/'pipeline_complete.json',{'best_checkpoint':str(best),'finished':time.strftime('%Y-%m-%dT%H:%M:%S')})
    print('FARE_V4_PIPELINE_COMPLETE',flush=True)
    return run_dir
