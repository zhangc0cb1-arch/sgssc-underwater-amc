# No-added-noise training

The package provides the two SGSSC stages without synthetic-noise augmentation.
Training and validation arrays are **not distributed**. The released test NPZ
files are for evaluation only and are not training inputs.

## External data directories

Each directory is one declared split. Arrays preserve the source order and
are loaded with `allow_pickle=False`; I/Q arrays are memory-mapped.

```text
training_data/
  pretrain_train/   # full unlabeled training pool
    manifest.json
    iq.npy
    base_ids.npy
  labeled_train/    # preselected labeled subset of that training pool
    manifest.json
    iq.npy
    base_ids.npy
    labels.npy
  val/             # separate validation observations
    manifest.json
    iq.npy
    base_ids.npy
    labels.npy
```

| File | Required contents |
|---|---|
| `iq.npy` | float32 `[N,512,2]`, complete unmasked I/Q; finite values |
| `base_ids.npy` | Unicode `[N]`, nonempty unique source identities |
| `labels.npy` | int64 `[N]`, class IDs 0 through 10; classification only |

Example `manifest.json` for either training directory:

```json
{
  "schema": "sgssc-clean-data-v1",
  "split": "train",
  "noise_mode": "none_added",
  "class_names": ["BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64", "GFSK", "CPFSK", "AM-DSB", "AM-SSB", "WBFM"]
}
```

Use `"split": "val"` in the validation manifest. Pretraining never opens
`labels.npy`, even if it is present. Classification requires labels in both
its training subset and validation set.

Prepare no-added-noise inputs at the same sampling rate and scale as the model.
For the historical source data, this means 8 kHz, 512 complex samples, and the
already applied train-fitted scale `3.0439862190625893`. Do not scale those
historical arrays again. The loader neither fits a scale nor adds noise.
For a new dataset, fit preprocessing on training data and reuse it for other
splits; declare its identity and preprocessing in your own data records.

Training/validation base IDs must not overlap. Test split declarations,
directories named with a `test` prefix, and IDs with a `test` prefix are
rejected. The manifest is a user declaration; it cannot prove physical
channel independence or detect renamed test observations. Preserve the
original grouping, temporal exclusions, and test holdout when preparing data.

## Stage 1: masked spectrum pretraining

```bash
python scripts/train_clean.py pretrain --train-data training_data/pretrain_train --val-data training_data/val --output training_runs/pretrain
```

Defaults match the historical clean protocol: 20,000 batch updates, batch 64,
Adam with learning rate `1e-4` and no weight decay, seed `20261001`, and four CPU
threads. Each update samples with replacement, masks 70% of time points in
both I/Q channels together, and minimizes MSE against the complete shifted,
orthonormal complex FFT. Both encoder and spectrum head receive gradients.

The validation subset contains 512 observations by default, selected once
without replacement. Its per-ID masks stay fixed. Validation occurs at step
0, every 250 updates, and the final step; the lowest MSE wins, with the
earliest checkpoint retained on ties. Validation labels are not used.

`--validation-observations` permits smaller separately supplied validation
sets. `--model-config` can read a JSON object of `SGSSCConfig` fields for a
different architecture. The released weights use the default architecture.

## Stage 2: frozen-encoder classifier training

```bash
python scripts/train_clean.py classifier --train-data training_data/labeled_train --val-data training_data/val --pretrained training_runs/pretrain/best.pt --output training_runs/classifier
```

Alternatively use `--pretrained weights/sgssc_inference.pt` after extracting
the release assets. The existing classifier is not reused: its attention
readout and final linear layer are initialized anew. Only the encoder and
spectrum head are loaded from the pretrained source.

The supplied `labeled_train` directory must already contain the selected
labeled subset. The default count is **1,210**, checked as
`round(242000 * 0.005)`. `--training-pool-size` and `--label-fraction` describe
other externally selected subsets. The trainer does not select labels using
test data or classifier results. Reusing the historical run requires its
original labeled-subset IDs and row order.

Defaults: 2,000 batch updates, batch 64, Adam `1e-3`, seed `20261001`, and four
CPU threads. The encoder is frozen in evaluation mode; its layer features
are cached. Only the readout is optimized with cross-entropy, using complete
I/Q without masking or PCA. The full validation split is evaluated at step 0,
every 250 updates, and the final step. The highest correct count wins, with
earliest ties. Encoder weights are checked for changes during training.

Pretrained checkpoints must declare `noise_mode=none_added`. The original
released checkpoint predates that field and is accepted only by its verified
SHA-256 identity; an undeclared arbitrary checkpoint is not assumed clean.

## Outputs and checks

Outputs go into a **new directory**; existing directories are refused.
Neither stage overwrites the published checkpoint or input arrays.

Both stages write `run.json`, `report.json`, `curve.jsonl`, `best.pt`, `last.pt`,
and `training_exposures.npz`. Classifier training additionally writes feature
caches and the best validation predictions. The reports include data hashes,
validation selection, seed, settings, and source checkpoint identity.
Checkpoint loading uses `weights_only=True`.

The classifier's `best.pt` is accepted by the evaluation and visualization CLI:

```bash
python -m sgssc_amc classify --checkpoint training_runs/classifier/best.pt --output results/new_classifier
python -m sgssc_amc visualize --checkpoint training_runs/classifier/best.pt --output results/new_classifier_features
```

`--steps`, `--lr`, `--batch-size`, `--validate-every`, `--seed`, and `--threads`
are available on both training commands. These counts are updates, not epochs.
Training outputs contain optimizer/RNG state for inspection; this version
does not expose a resume command.

The packaged training code was checked with small synthetic examples for
gradients, frozen encoder behavior, split rejection, fixed masks, and safe
checkpoint reload. No full-budget retraining was performed while preparing
this release. Architecture and objective match the historical implementation;
different data order, software, or numerical execution can change a new run.
