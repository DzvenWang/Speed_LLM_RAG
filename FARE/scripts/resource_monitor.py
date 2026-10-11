#!/usr/bin/env python3
"""Job-scoped, low-overhead GPU/CPU/RSS sampling for FARE PBS jobs.

The monitor is best-effort; nvidia-smi or /proc failures become null fields,
not training failures. Samples are appended immediately to JSONL and CSV.
On normal SIGTERM the summary is written; abrupt walltime cancellation still
leaves all prior samples available for offline summarisation with --summarize.
Only a GPU with a compute PID in the monitored process tree is attributed to
this job, rather than accidentally reporting another job's device usage.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

FIELDS = ('timestamp','unix_time','elapsed_seconds','process_count','cpu_cores',
          'process_rss_gb','process_read_gb','process_written_gb',
          'gpu_index','gpu_uuid','gpu_util_pct','gpu_memory_util_pct',
          'gpu_memory_used_mb','gpu_memory_total_mb','job_gpu_process_memory_mb',
          'gpu_power_w','gpu_temperature_c','gpu_matched','nvidia_smi_ok')


def _float(text):
    value=str(text).strip()
    if not value or value.lower() in ('n/a','[not supported]','not supported','none'):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _command(arguments,timeout=6):
    try:
        result=subprocess.run(arguments,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                              text=True,timeout=timeout,check=False)
        return result.stdout if result.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):
        return None


def _proc_ppid(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
        fields=raw[raw.rfind(')')+2:].split()
        return int(fields[1])
    except (OSError,ValueError,IndexError):
        return None


def _proc_pids(root):
    parents={}
    try:
        for entry in os.scandir('/proc'):
            if entry.name.isdigit():
                pid=int(entry.name)
                ppid=_proc_ppid(pid)
                if ppid is not None:parents[pid]=ppid
    except OSError:
        return set()
    family={int(root)}
    for _ in range(12):
        new={pid for pid,ppid in parents.items() if ppid in family}
        if new <= family:break
        family |= new
    return family


def _process_snapshot(pids):
    ticks=os.sysconf('SC_CLK_TCK')
    pages=os.sysconf('SC_PAGE_SIZE')
    cpu_ticks=0
    rss_bytes=0
    read_bytes=0
    write_bytes=0
    found=0
    for pid in pids:
        try:
            text=Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
            f=text[text.rfind(')')+2:].split()
            cpu_ticks+=int(f[11])+int(f[12])  # utime, stime
            rss_bytes+=max(0,int(f[21]))*pages  # rss pages
            found+=1
            try:
                lines=Path(f'/proc/{pid}/io').read_text(encoding='utf-8').splitlines()
                io=dict(line.split(':',1) for line in lines if ':' in line)
                read_bytes+=int(io.get('read_bytes',0))
                write_bytes+=int(io.get('write_bytes',0))
            except (OSError,ValueError):
                pass
        except (OSError,ValueError,IndexError):
            continue
    return {'count':found,'cpu_seconds':cpu_ticks/ticks,'rss_gb':rss_bytes/1e9,
            'read_gb':read_bytes/1e9,'written_gb':write_bytes/1e9}


def _parse_gpu_table(text):
    if text is None:return []
    devices=[]
    for line in text.splitlines():
        row=[v.strip() for v in line.split(',')]
        if len(row)<8:continue
        devices.append({'index':row[0],'uuid':row[1],
                        'util':_float(row[2]),'mem_util':_float(row[3]),
                        'mem_used':_float(row[4]),'mem_total':_float(row[5]),
                        'power':_float(row[6]),'temp':_float(row[7])})
    return devices


def _parse_gpu_apps(text,pids):
    matching={}
    if text is None:return matching
    for line in text.splitlines():
        row=[v.strip() for v in line.split(',')]
        if len(row)<3:continue
        try:pid=int(row[0])
        except ValueError:continue
        if pid in pids:
            uuid=row[1]
            mem=_float(row[2])
            matching[uuid]=matching.get(uuid,0.) + (mem or 0.)
    return matching


def gather(parent_pid,start,prior):
    now=time.time()
    pids=_proc_pids(parent_pid)
    proc=_process_snapshot(pids)
    elapsed=now-start
    cpu_cores=None
    if prior is not None:
        dt_sec=now-prior[0]
        if dt_sec>0:cpu_cores=max(0.,(proc['cpu_seconds']-prior[1])/dt_sec)
    gpu_text=_command(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu',
                       '--format=csv,noheader,nounits'])
    apps_text=_command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                        '--format=csv,noheader,nounits'])
    devices=_parse_gpu_table(gpu_text)
    apps=_parse_gpu_apps(apps_text,pids)
    # Never silently attribute an unrelated GPU to the current PBS job.
    matched=[gpu for gpu in devices if gpu['uuid'] in apps]
    gpu=matched[0] if matched else None
    row={'timestamp':dt.datetime.now(dt.timezone.utc).isoformat(),
         'unix_time':round(now,3),'elapsed_seconds':round(elapsed,2),
         'process_count':proc['count'],'cpu_cores':round(cpu_cores,3) if cpu_cores is not None else None,
         'process_rss_gb':round(proc['rss_gb'],3),
         'process_read_gb':round(proc['read_gb'],3),
         'process_written_gb':round(proc['written_gb'],3),
         'gpu_index':gpu['index'] if gpu else None,
         'gpu_uuid':gpu['uuid'] if gpu else None,
         'gpu_util_pct':gpu['util'] if gpu else None,
         'gpu_memory_util_pct':gpu['mem_util'] if gpu else None,
         'gpu_memory_used_mb':gpu['mem_used'] if gpu else None,
         'gpu_memory_total_mb':gpu['mem_total'] if gpu else None,
         'job_gpu_process_memory_mb':apps.get(gpu['uuid']) if gpu else None,
         'gpu_power_w':gpu['power'] if gpu else None,
         'gpu_temperature_c':gpu['temp'] if gpu else None,
         'gpu_matched':bool(gpu),'nvidia_smi_ok':gpu_text is not None}
    return row,(now,proc['cpu_seconds'])


def _percentile(values,p):
    if not values:return None
    values=sorted(values)
    pos=(len(values)-1)*p
    lo=int(pos);hi=min(lo+1,len(values)-1)
    return round(values[lo]*(hi-pos+lo)+values[hi]*(pos-lo),3)


def summarize(rows):
    out={'sample_count':len(rows),'matched_gpu_samples':sum(bool(r.get('gpu_matched')) for r in rows),
         'units':{'gpu_memory':'MiB as reported by nvidia-smi','process_rss':'decimal GB; summed RSS may share pages',
                  'cpu_cores':'process CPU seconds per wall second; 1.0 = one full CPU core'},
         'notes':['GPU utilization is sampled device-wide, and only attributed when a matching job PID is found.',
                  'Short GPU kernels between samples may be missed.',
                  'CPU/RSS excludes child processes that already exited before a sample.']}
    if rows:
        out['elapsed_seconds']=round(float(rows[-1]['elapsed_seconds']),2)
    for key in ('gpu_util_pct','gpu_memory_util_pct','gpu_memory_used_mb',
                'job_gpu_process_memory_mb','gpu_power_w','gpu_temperature_c',
                'cpu_cores','process_rss_gb'):
        values=[float(r[key]) for r in rows if r.get(key) is not None]
        out[key]={'count':len(values),'mean':round(statistics.fmean(values),3) if values else None,
                  'p95':_percentile(values,.95),
                  'max':round(max(values),3) if values else None}
    return out


def read_jsonl(path):
    if not Path(path).exists():return []
    rows=[]
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent-pid',type=int)
    parser.add_argument('--output-prefix',required=True)
    parser.add_argument('--interval',type=float,default=30.)
    parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args(argv)
    prefix=Path(args.output_prefix)
    prefix.parent.mkdir(parents=True,exist_ok=True)
    jsonl=Path(f'{prefix}.jsonl');csvfile=Path(f'{prefix}.csv');summary=Path(f'{prefix}.summary.json')
    if args.summarize:
        summary.write_text(json.dumps(summarize(read_jsonl(jsonl)),indent=2),encoding='utf-8')
        return 0
    if args.parent_pid is None:parser.error('--parent-pid is required for monitoring')
    if args.interval<1:parser.error('--interval must be >= 1')
    stop=False
    def _signal(_signum,_frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,_signal)
    signal.signal(signal.SIGINT,_signal)
    start=time.time();prior=None
    with jsonl.open('a',encoding='utf-8',buffering=1) as jl, csvfile.open('a',encoding='utf-8',newline='') as cf:
        writer=csv.DictWriter(cf,fieldnames=FIELDS)
        if cf.tell()==0:writer.writeheader()
        while not stop:
            row,prior=gather(args.parent_pid,start,prior)
            jl.write(json.dumps(row,allow_nan=False)+'\n')
            writer.writerow(row);cf.flush()
            if not Path(f'/proc/{args.parent_pid}').exists():break
            stop_at=time.monotonic()+args.interval
            while not stop and time.monotonic()<stop_at:
                time.sleep(min(.5,max(0.,stop_at-time.monotonic())))
    summary.write_text(json.dumps(summarize(read_jsonl(jsonl)),indent=2),encoding='utf-8')
    return 0

if __name__=='__main__':sys.exit(main())
