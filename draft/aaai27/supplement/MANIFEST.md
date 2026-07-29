# Anonymous Code and Data Supplement Manifest

This file defines the intended AAAI-27 Code and Data Supplement. The final ZIP
must be self-contained, anonymous, and free of links to an identifying public
repository.

## Required Top-Level Contents

```text
README.md
LICENSES.md
environment/
  environment.yml
  pip-freeze.txt
  system.txt
configs/
data/
  train/
  validation/
  ood/
predictions/
  id/
  ood/
  capability/
checkpoints/
  manifests/
results/
  estimates/
  robust_inference/
  tables/
src/
  train/
  eval/
scripts/
  reproduce_table_1.sh
  reproduce_table_2.sh
  reproduce_capabilities.sh
```

## Claim-Carrying Inventory

| Artifact | Required | Current status |
|---|---:|---|
| Exact local base ID raw X/Y predictions | yes | not staged |
| Qwen-own selected seed raw ID X/Y predictions | yes | not staged |
| Exact local base OOD raw X/Y predictions | yes | not staged |
| Qwen-own selected seed OOD raw X/Y predictions | yes | not staged |
| Consensus ablation raw ID/OOD predictions | yes | not staged |
| SFT raw ID/OOD predictions | yes | run pending |
| Seed 1 and seed 2 raw ID/OOD predictions | yes | evaluations complete; raw OOD records not staged |
| NLS/robust-inference outputs | yes | derived NLS partial; robust pending |
| Adapter weights or anonymous retrievable archive | yes | checksums only |
| Dataset files and freeze hashes | yes | OOD derived files exist on origin/main; staging pending |
| Checkpoint-selection records for every seed | yes | seed 42, seed 1, and seed 2 records committed in project results |
| GSM8K predictions and paired comparison | yes | summaries exist; raw predictions not staged |
| IFEval predictions and scorer output | yes | pending |
| Table/figure generation scripts | yes | paper table generator started |
| Exact commands for every primary table | yes | pending consolidated README |

## Anonymity Scrub

Before zipping:

- remove author names, usernames, home/scratch paths, institution names, email
  addresses, grant identifiers, repository remotes, and identifying URLs;
- rewrite absolute paths in metadata and logs;
- remove Git history and `.git` directories;
- remove scheduler account/project names from PBS logs;
- inspect PDF, PNG, JSON, notebook, ZIP, and model metadata;
- ensure filenames do not contain author surnames or account names;
- run the supplement in a clean directory with no access to the original repo.

## Reproduction Contract

The staged package is complete only when a reviewer can:

1. reconstruct every row of the primary ID and OOD tables from included raw
   predictions;
2. reproduce checkpoint selection from the declared validation grid;
3. reproduce GSM8K and IFEval comparisons from included predictions;
4. inspect all final configs, seeds, model/checkpoint hashes, and environment
   versions; and
5. run one command per primary table without accessing an external private or
   identifying resource.
