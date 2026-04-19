# Documents Folder

## Why This Folder Exists

`Documents/` centralizes human-readable technical documentation so contributors can understand design intent, implementation details, and benchmark context without reverse-engineering code first.

## What This Folder Contains

```text
Documents/
  Architecture.md
  Code.md
  Benchmark_Reference.md
```

## File Guide

## 1) `Architecture.md`

Use this when you need:

- system-level mental model,
- component boundaries,
- data flow across training/inference,
- folder-level architecture rationale.

## 2) `Code.md`

Use this when you need:

- implementation-level explanation,
- RL concepts tied to this project,
- actor-critic and PPO mechanics as implemented,
- operational interpretation of outputs/metrics.

## 3) `Benchmark_Reference.md`

Use this when you need:

- known benchmark/BKS references for X instances,
- context for score comparison quality,
- external reference values while interpreting results.

## Recommended Reading Order

1. `../README.md`
2. `Architecture.md`
3. `Code.md`
4. `Benchmark_Reference.md`

Then jump into code:

- `../Model/README.md`
- `../Baseline/README.md`
- `../Backend/README.md`

## How This Folder Fits the Running Mechanism

- Before running experiments: read architecture/code docs to set expectations.
- During debugging: cross-check implementation behavior and metric definitions.
- During result review: compare observed score behavior with benchmark references.

## Documentation Maintenance Policy

Whenever behavior changes in any of these areas, update docs in the same change-set:

- action space,
- reward logic,
- best-model selection metric,
- default output locations and file naming,
- training/evaluation protocol.

This keeps docs trustworthy for teammates and reviewers.
