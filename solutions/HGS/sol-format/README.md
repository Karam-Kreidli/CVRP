# Solutions/HGS/sol-Format Folder

## Why This Folder Exists

This folder stores `.sol` route files for baseline HGS runs.

## File Format

Each file is plain text, one route per line:

```text
Route #1: 35 46 31
Route #2: 15 22 41 20
```

- filename pattern: `<instance>.sol`
- route IDs are 1-based
- node IDs are customer indices used by solver output conventions

## Typical Producer

- `Baseline/Portfolio_Solver.py`
