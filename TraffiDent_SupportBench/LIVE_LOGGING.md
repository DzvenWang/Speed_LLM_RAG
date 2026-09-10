# Real-time logging (v1.2.2)

PBS uses `python -u`, `PYTHONUNBUFFERED=1` and `timestamp_stream.py | tee` so every stdout/stderr line is flushed to `logs/SupportBenchBuild.<JOBID>.live.log` with one timestamp. `build_all.sh` prints START/DONE/FAILED plus elapsed seconds for every stage.
