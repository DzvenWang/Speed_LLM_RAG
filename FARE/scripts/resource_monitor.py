#!/usr/bin/env python3
"""Lightweight PBS process/allocated GPU telemetry; no extra pip dependency."""
import argparse,csv,json,os,subprocess,time,signal
from pathlib import Path

STOP=False
signal.signal(signal.SIGTERM,lambda *_:globals().__setitem__('STOP',True))
signal.signal(signal.SIGINT,lambda *_:globals().__setitem__('STOP',True))

def alive(pid):return Path(f'/proc/{pid}').exists()

def gpu_stats():
    try:
        s=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total',
                                   '--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()[0]
        util,mem,total=(float(x.strip()) for x in s.split(',')[:3]);return util,mem,total
    except Exception:return None,None,None

def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):return int(line.split()[1])/1024
    except Exception:pass
    return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-pid',type=int,required=True)
    p.add_argument('--interval',type=int,default=30);p.add_argument('--output-prefix',required=True)
    a=p.parse_args();path=Path(a.output_prefix+'.csv');path.parent.mkdir(parents=True,exist_ok=True)
    stats=[]
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['timestamp','gpu_util_percent','gpu_mem_mib','gpu_total_mib','parent_rss_mib'])
        writer.writeheader();f.flush()
        while not STOP and alive(a.parent_pid):
            util,mem,total=gpu_stats();row={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),
                'gpu_util_percent':util,'gpu_mem_mib':mem,'gpu_total_mib':total,
                'parent_rss_mib':rss(a.parent_pid)}
            writer.writerow(row);f.flush();stats.append(row)
            for _ in range(max(a.interval,1)):
                if STOP or not alive(a.parent_pid):break
                time.sleep(1)
    valid=[x for x in stats if x['gpu_util_percent'] is not None]
    summary={'rows':len(stats),'valid_gpu_samples':len(valid),
             'mean_gpu_util_percent':sum(x['gpu_util_percent'] for x in valid)/len(valid) if valid else None,
             'max_gpu_mem_mib':max((x['gpu_mem_mib'] for x in valid),default=None)}
    Path(a.output_prefix+'.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
