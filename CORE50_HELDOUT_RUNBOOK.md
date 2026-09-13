# CORe50 NICv2-79 paper protocol

This runbook describes the CORe50 protocol reported in manuscript Table III.
It preserves the batch and temporal ordering of the official NICv2-79 stream
and evaluates held-out classes with the same MAML adaptation framework used by
Omniglot and CIFAR100.

## Protocol

- The 50 objects are divided into 30 mutually disjoint meta-training classes
  and 20 meta-test classes.
- The 20 unseen meta-test classes form four consecutive 5-way tasks.
- Meta-train support sessions: 1, 2, 4, 5.
- Meta-train query sessions: 6, 8, 9, 11.
- Validation/test adaptation uses samples from the official training sessions.
- Validation/test query uses only official test sessions 3, 7, 10.
- Five-way, two-shot support gives the ten inner updates inherited from the
  Omniglot/CIFAR100 MAML configuration.
- Model selection is completed before final meta-test evaluation; the final
  four-task stream is evaluated once with frozen hyperparameters.
- During meta-training, held-out validation runs every 1000 iterations by
default and saves `<run>_best.pt` by validation ACC. Validation RNG state is
restored afterward so evaluation does not change the training trajectory.
- The matching `<run>_validation_best.json` records both the selected training
  iteration and inner learning rate for the one-shot meta-test command.
- The 20 meta-test classes are never evaluated during meta-training.
- Replay is used only during FSEML-ER decoder meta-training. Validation and
  meta-test adaptation are replay-free for both methods.
- Replay partitions are built from the 30 meta-training classes only; the 20
  held-out classes never enter the meta-training replay buffer.

Generate the immutable split manifest:

```bash
python prepare_core50_heldout_split.py \
  --seed 9 --output core50_heldout_splits/split_seed9.json
```

## Smoke test

```bash
python train_core50_heldout.py \
  --data-root "$CORE50_DATA_ROOT" --split core50_heldout_splits/split_seed9.json \
  --seed 9 --meta-iterations 10 --no-replay \
  --output-dir core50_heldout_smoke
```

## Meta-train

FSEML:

```bash
nohup env CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-core50-heldout \
python -u train_core50_heldout.py --data-root "$CORE50_DATA_ROOT" \
--split core50_heldout_splits/split_seed9.json --seed 9 \
--meta-iterations 5000 --ways 5 --support-shots 2 --query-shots 5 \
--learning-rate 0.0001 --inner-lr 0.05 --update-step 10 --no-replay \
--validation-every 1000 --checkpoint-every 1000 \
--output-dir core50_heldout_fseml > ../log/core50_heldout_fseml_seed9.log 2>&1 &
```

FSEML-ER-decoder:

```bash
nohup env CUDA_VISIBLE_DEVICES=1 MPLCONFIGDIR=/tmp/matplotlib-core50-heldout \
python -u train_core50_heldout.py --data-root "$CORE50_DATA_ROOT" \
--split core50_heldout_splits/split_seed9.json --seed 9 \
--meta-iterations 5000 --ways 5 --support-shots 2 --query-shots 5 \
--learning-rate 0.0001 --inner-lr 0.05 --update-step 10 \
--replay --replay-content decoder --replay-warmup 100 \
--buffer-size 1000 --replay-gap 32 --replay-rate 1.0 --replay-partitions 50 \
--validation-every 1000 --checkpoint-every 1000 \
--output-dir core50_heldout_fseml_er > ../log/core50_heldout_fseml_er_seed9.log 2>&1 &
```

## Training-side selection and meta-test

Select the checkpoint and learning rate using training-side validation data;
the 20 unseen meta-test classes must not be used for this selection. Example
for FSEML:

```bash
python evaluate_core50_heldout.py \
  --checkpoint core50_heldout_fseml/core50_heldout_fseml_seed9.pt \
  --data-root "$CORE50_DATA_ROOT" --split core50_heldout_splits/split_seed9.json \
  --phase validation --inner-lrs 0.01 0.05 0.1 \
  --prediction-class-scope task \
  --output core50_heldout_fseml/validation_seed9.json
```

Freeze the selected learning rate before running the four-task meta-test:

```bash
python evaluate_core50_heldout.py \
  --checkpoint core50_heldout_fseml/core50_heldout_fseml_seed9.pt \
  --data-root "$CORE50_DATA_ROOT" --split core50_heldout_splits/split_seed9.json \
  --phase test --selection core50_heldout_fseml/validation_seed9.json \
  --prediction-class-scope task \
  --output core50_heldout_fseml/test_seed9.json
```

Repeat those two commands with the FSEML-ER checkpoint and output directory.
Report task-aware final ACC, learning ACC, and forgetting. A matched
`--prediction-class-scope seen` evaluation may be included as a stricter
supplement but must not be mixed with the task-aware main result.
