# Sources and attribution

## SGSSC research method

The architecture and pretext task are based on **Predicting Spectral Information for Self-Supervised Signal Classification**, IJCAI 2025:

- Paper: <https://www.ijcai.org/proceedings/2025/0752.pdf>

This package is an independent implementation and adaptation for underwater acoustic signals. It is not the paper authors' official implementation. The 512-sample input, joint I/Q patches, ordered flatten-to-linear spectrum head, data protocol, and training budgets include project-specific implementation choices. The paper itself is not redistributed in this package.

## Underwater Acoustic Channel Repository: Pink data

The included test signals are derived from the **Pink** channel impulse responses in:

**Zhengnan Li, Mandar Chitre, Diego Cuji, James Preisig, Andrew Singer, Milica Stojanovic, and Paul van Walree. Underwater Acoustic Channel Repository, version 1.0 (2026).**

- Version DOI: <https://doi.org/10.5281/zenodo.21287414>
- Record: <https://zenodo.org/records/21287414>
- Pink channel description: <https://uwa-channels.github.io/channels/pink/>
- Upstream data license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, <https://creativecommons.org/licenses/by/4.0/>

The source record's author list, version, DOI, and license were checked against the archived official Zenodo metadata. Original Pink CIR files are not included.

**Derivation and changes:** synthetic complex-baseband modulation signals were replayed through the measured channels, filtered and cropped to 512 complex samples at 8 kHz, and processed using the source project's fixed training-derived scale. This release exports only the existing held-out test windows and their metadata. It also derives fixed masked inputs and orthonormal complex FFT targets for spectrum evaluation. These arrays are semi-synthetic test data, not direct recordings of labeled modulation transmissions at sea. Preserve this source attribution, the CC BY 4.0 notice, and the derivation statement when redistributing the data.

## Channel replay software used during data preparation

The source research project used **UWA Channels Python**, version **0.7.1**, commit `1b55abbcf07e2893c7d78edf1e284b906380a3b5`:

- Source: <https://github.com/uwa-channels/python>
- Copyright (c) 2025–2026 Underwater Acoustic Channels Group
- License: MIT

The replay package is not required to evaluate the exported data. Its upstream license is reproduced below for attribution; it does not license this project's original code or model weights.

```text
MIT License

Copyright (c) 2025-2026 Underwater Acoustic Channels Group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Python dependencies

NumPy, PyTorch, and Matplotlib are installed separately and remain subject to their respective upstream licenses. Optional export tooling uses h5py. Optional feature visualization uses scikit-learn and threadpoolctl, with scikit-learn's NumPy, SciPy, and joblib dependencies. Their source distributions are not vendored here.

## This package's code and weights

The project author has not yet selected a distribution license for the original code and model weights. No project-wide MIT or other open-source license is asserted. The data attribution and upstream license notices above remain separate.

