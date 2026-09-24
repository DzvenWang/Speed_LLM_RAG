# Windows notes

P1 is path-portable. By default the workspace is the parent directory of `PreSuRe`.

```powershell
$env:PRESURE_WORKSPACE_ROOT="C:\Users\zwan0652\ResearchProgram\Speed_LLM_RAG"
python scripts\run_p1.py --stage development
```

`runtime.num_workers=0` is the default for Windows stability. The frozen PatchSTG candidate/history branch is precomputed once, so adapter epochs do not rerun candidate PatchSTG inference.
