# Paper table, command, and result mapping

`expected_results.json` lists the expected metrics in percentage points.

| Manuscript table | Row | Training | Evaluation | Expected-result ID |
|---|---|---|---|---|
| I | Omniglot FSEML / FSEML-ER | `run_paper_five_runs.py --dataset omniglot` | `evaluate_omniglot_continual.py` | `table-i-*` |
| II | CIFAR100 FSEML / FSEML-ER, 5/10/20 classes | command A with matching `N` | command B | `table-ii-*` |
| III | CORe50 NICv2-79 FSEML / FSEML-ER | `CORE50_HELDOUT_RUNBOOK.md` | `evaluate_core50_heldout.py` | `table-iii-core50-fseml*` |
| III | CLINC150 FSEML / FSEML-ER | `CLINC150_RUNBOOK.md` | `evaluate_clinc150_heldout.py` | `table-iii-clinc150-fseml*` |

Only these FSEML/FSEML-ER rows in Tables I–III are in the public package's
required reproduction scope. Tables IV–IX are supporting analyses and are not
part of the one-command acceptance criterion.

Command A (set `N` to 5, 10, or 20):

```bash
python mrcl_classification.py --dataset CIFAR100 \
  --data-root "$CIFAR100_DATA_ROOT" --classes-per-task N \
  --task-protocol cifar100_protocols/cifar100_Nclasses_seed9.json \
  --steps 20000 --seed 9 --meta_lr 0.0005 --update_lr 0.1 \
  --weight-decay-weight 0.1 --no-replay --model_name "$CHECKPOINT"
```

For FSEML-ER replace `--no-replay` with:

```text
--replay --replay-content decoder --replay-rate 0.05 --replay-gap 960 --replay-buffer-size 1000
```

Command B:

```bash
python evaluate_cifar100_continual.py --model "$CHECKPOINT" \
  --data-root "$CIFAR100_DATA_ROOT" \
  --task-protocol cifar100_protocols/cifar100_Nclasses_seed9.json \
  --classes-per-task N --seed 9 --prediction-class-scope task \
  --run-name cifar100_N_fseml
```

The manuscript reports five independent PyTorch models and five runs with
different class batches.
`$CHECKPOINT` is user-selected because checkpoints are separate release assets.

The complete five-run configuration is `configs/paper_main_five_runs.json`.
The seed set is 9/19/29/39/49, and every method uses the same split for a given
seed. Generate or verify the 15 CIFAR protocol files with:

```bash
python3 prepare_cifar100_protocol.py --seeds 9 19 29 39 49
```

Preview all commands safely, then execute the requested subset:

```bash
python3 run_paper_five_runs.py
python3 run_paper_five_runs.py --dataset cifar100 --method fseml --execute
python3 run_paper_five_runs.py --dataset omniglot --method fseml-er --execute
```

For every Omniglot checkpoint the runner first invokes
`evaluate_omniglot_continual.py --phase select` on background/meta-training
classes. It writes the selected learning rate to a seed-specific JSON file.
The subsequent `--phase test` command requires that file, freezes the selected
rate, processes all 660 unseen one-class tasks in deterministic seed order, and
writes both the full accuracy matrix and ACC/FM/LA metrics. No
meta-test query accuracy participates in model or learning-rate selection.

After the Omniglot and CIFAR runs finish, aggregate all five seeds and compare
their means with Tables I–II:

```bash
python3 summarize_paper_five_runs.py
python3 verify_results.py --actual paper_runs/cifar100_five_run_summary.json
```

Preview or execute all four Table III pipelines:

```bash
python3 run_paper_table_iii.py
python3 run_paper_table_iii.py --execute
python3 summarize_paper_table_iii.py
python3 verify_results.py --actual paper_runs/table_iii_summary.json
```

The Table III runner defaults to seeds 9/19/29/39/49. Its summarizer requires
all five outputs for each dataset/method row and reports mean ± sample standard
deviation.

Validate the manuscript result specification or compare reproduced metrics:

```bash
python3 verify_results.py
python3 verify_results.py --actual reproduced_metrics.json
```

The optional actual file maps a manuscript result ID to scalar metrics, e.g.
`{"table-ii-fseml-c5":{"ACC":77.3,"FM":2.2,"LA":80.0}}`. Comparison uses
the reported standard deviation when present, otherwise `--atol` (default
0.01 manuscript units).
