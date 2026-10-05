from __future__ import annotations

from pathlib import Path


def portable_project_path(path: str | Path, project_root: str | Path) -> str:
    p = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    try:
        return p.relative_to(root).as_posix()
    except ValueError as e:
        raise RuntimeError(f'Pointer target must live under project root {root}: {p}') from e


def resolve_portable_pointer(raw: str | Path, project_root: str | Path, *, pointer_parent: str | Path | None = None) -> Path:
    text = str(raw).strip()
    if not text:
        raise ValueError('Empty pointer')
    normalized = text.replace('\\', '/')
    root = Path(project_root).expanduser().resolve()
    candidates: list[Path] = []
    p = Path(normalized).expanduser()
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(root / p)
        if pointer_parent is not None:
            candidates.append(Path(pointer_parent).expanduser().resolve() / p)
    parts=[x for x in normalized.split('/') if x]
    for anchor in ('Backbone_exps','Baseline_exps'):
        if anchor in parts:
            i=parts.index(anchor)
            candidates.append(root.joinpath(*parts[i:]))
    seen=set()
    for c in candidates:
        try: cr=c.resolve()
        except Exception: cr=c
        key=str(cr)
        if key in seen: continue
        seen.add(key)
        if cr.exists(): return cr
    if p.is_absolute():
        for anchor in ('Backbone_exps','Baseline_exps'):
            if anchor in parts:
                i=parts.index(anchor)
                return root.joinpath(*parts[i:]).resolve()
        return p
    return (root/p).resolve()
