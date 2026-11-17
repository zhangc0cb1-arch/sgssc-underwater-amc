# Held-out test data

This release contains **test data only**. No training or validation waveforms are
included. Both tasks use the same complete 2,200-example `test_seen` split and
2,200-example `test_unseen` split, in the original classification evaluation's
base-ID order. There is no selection based on model predictions.

The data are semi-synthetic: generated modulation signals were replayed through
measured Pink underwater-acoustic channel impulse responses. They are not
real-world recordings of the eleven modulation classes.

## Classification

Files: `classification/test_seen.npz` and `classification/test_unseen.npz`.

| Array | Shape | Type | Meaning |
|---|---|---|---|
| `iq` | `(2200, 512, 2)` | float32 | Complete received reference; last axis is I, Q |
| `labels` | `(2200,)` | int64 | Modulation class, 0 through 10 |
| `base_ids` | `(2200,)` | Unicode string | Frozen example identity |

Class order: BPSK, QPSK, 8PSK, PAM4, QAM16, QAM64, GFSK, CPFSK,
AM-DSB, AM-SSB, WBFM. Each split has 200 examples per class.

The accompanying `*_metadata.csv` files locate every example in its historical
source shard and describe source ID, channel record, receiver, replay start and
crop coordinates. Paths in these files are relative to the original research
archive; that archive is not required for inference on the exported NPZ files.

## Spectral reconstruction

Files: `spectrum/test_seen.npz` and `spectrum/test_unseen.npz`.

| Array | Shape | Type | Meaning |
|---|---|---|---|
| `masked_iq` | `(2200, 512, 2)` | float32 | Received reference with the fixed mask applied |
| `target_spectrum` | `(2200, 512, 2)` | float32 | Full unmasked reference's complex DFT, real/imaginary |
| `masked_positions` | `(2200, 512)` | bool | `True` means masked; both I/Q coordinates are zeroed |
| `base_ids` | `(2200,)` | Unicode string | Same identities/order as the classification split |

Each example has exactly 358 masked time points, with requested mask ratio 0.7
and actual ratio 358/512. The seed is **20261117**. Seeds are derived independently
from each base ID using the archived stable-seed and validation-mask algorithms;
the historical `sgssc-val-mask` string namespace is retained, but these examples
are held-out **test** examples. Stored masks are independent of batch order and
any model output. The complete algorithm and software versions are recorded in
`../provenance/data_manifest.json`.

Targets use `torch.fft.fft(..., norm="ortho")`, followed by `fftshift`, across all
512 complex samples. They contain all frequency bins in negative-first order;
the two output coordinates are real and imaginary parts, not magnitude and
phase. No class labels are included in the spectrum archives.

**This fixed spectral test task was created for this release from the already
held-out waveforms.** It is not a historical pretraining evaluation result.
The archived pretraining report's 512 examples were validation monitoring and
are not included here. Results obtained on this new test task must be identified
as new evaluation results, separately from the historical classifier results.

## Signal scale and split isolation

Sampling rate is 8,000 Hz; each input is 512 complex samples. Inputs are exact
copies of the archived float32 `r` field, without additional noise, rescaling,
filtering, alignment or per-example normalization. The historical shared scale
is **3.0439862190625893**, fitted on training inputs only. Do not divide by it
again when loading these files. “No added noise” still includes channel
propagation and channel-estimation effects.

`test_seen` uses receiver 0 of pink_2/pink_3 in the held-out CIR interval
[44,56) seconds; training used [0,24), validation [28,40). `test_unseen` uses
receiver 0 of the entirely held-out pink_1 record. Treating each Pink file as an
independent channel group was a research assumption, not a verified statement
about physical acquisition independence. All test base IDs and source IDs were
checked against the 242,000 training and 2,200 validation identity manifests.

## Provenance and re-export

`../provenance/data_manifest.json` records source and exported checksums,
normalization, array shapes, splits, upstream attribution, and export checks.
Every input value was compared with a second read of its original HDF5 `r`
value; labels match the historical classification artifact; every serialized
array was checked after NPZ reload.

For maintainers who have the original research archive, re-export to an empty
destination using:

```bash
python scripts/export_test_data.py --source-root /path/to/original-uwa-amc --output-root /path/to/new-export
```

The exporter refuses to overwrite existing output artifacts. It does not train
or run a model, and reads training/validation metadata only to check isolation.
Ordinary users can load the bundled files directly with
`numpy.load(path, allow_pickle=False)`.

The source channel dataset is *Underwater Acoustic Channel Repository*, version
1.0, DOI [10.5281/zenodo.21287414](https://doi.org/10.5281/zenodo.21287414),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
See `../THIRD_PARTY_NOTICES.md` and the data manifest for complete attribution
and the transformations used to produce these derived examples.
