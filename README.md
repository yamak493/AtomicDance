# Music-to-Dance Generation via Atomic Movements

<!-- Replace the # targets below when the public resources are available. -->
[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red)](https://cxhcmhhh.github.io/AtomicDanceProject/static/pdfs/paper.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2607.13978-b31b1b.svg)](https://arxiv.org/abs/2607.13978)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=githubpages&logoColor=blue)](https://cxhcmhhh.github.io/AtomicDanceProject/)
[![Dataset](https://img.shields.io/badge/Google_Drive-Storage-dfa12b?style=flat&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1ETsaetMMWeKV3_E3Lr40BdybAsUAG8WM/view?usp=sharing)
[![YouTube](https://img.shields.io/badge/YouTube-Video-red?style=plastic&logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=gFabJjdnhdE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yamak493/AtomicDance/blob/main/colab/AtomicDance_Colab.ipynb)

This repository is the official PyTorch implementation of the paper
**Music-to-Dance Generation via Atomic Movements**.

**Xinhao Cai**, **Yixuan Sun**, **Minghang Zheng**, **Qingchao Chen**,
**Xin Jin**, **Song-Chun Zhu**, and **Yang Liu**

[Paper](https://cxhcmhhh.github.io/AtomicDanceProject/static/pdfs/paper.pdf) | [arXiv](https://arxiv.org/abs/2607.13978) | [Project](https://cxhcmhhh.github.io/AtomicDanceProject/) | [Dataset](https://drive.google.com/file/d/1ETsaetMMWeKV3_E3Lr40BdybAsUAG8WM/view?usp=sharing) | [YouTube](https://www.youtube.com/watch?v=gFabJjdnhdE)

Music-driven dance generation should produce motion that is rhythmically
synchronized with music while preserving coherent choreographic structure.
Existing end-to-end methods usually model dance as a continuous signal and
overlook its compositional nature. We instead represent choreography as a
sequence of semantically interpretable and reusable **atomic movements**.

We first construct an atomic movement vocabulary by segmenting dance sequences,
clustering recurring motion patterns, and refining their semantics with
LLM-assisted relabeling. We then introduce a two-stage generation framework
that mirrors the choreography process. A full-music-aware planner predicts the
type, timing, and duration of atomic movements. A transition-aware diffusion
model retrieves suitable movement prototypes, re-creates them with variations,
and synthesizes smooth, musically aligned transitions. The explicit symbolic
plan also enables users to replace movements, adjust durations, and edit dance
structure without retraining.

Our paper was accepted by ECCV 2026.

<!-- Add the public teaser/framework image here when available.
<div align="center">
  <img src="assets/teaser.png" width="90%">
</div>
-->

## Environment Setup

The project runs on two supported setups: the original Conda environment, and
Google Colab for anyone without a local GPU.

### Google Colab

Open [`colab/AtomicDance_Colab.ipynb`](colab/AtomicDance_Colab.ipynb) and select
a GPU runtime. It fetches the pretrained checkpoints and goes straight to
generation, so training is not required; sections 1-5 run in minutes. Training
is an optional section that mirrors checkpoints to Google Drive, so a
disconnected session can resume with `--resume`. Evaluation is optional too.

Colab ships its own CUDA build of PyTorch, so install the notebook's dependency
set rather than `requirements.txt`, whose CUDA 11.6 wheels would replace it:

```bash
pip install -r requirements-colab.txt
```

Neither PyTorch3D nor chumpy is needed there. PyTorch3D has no wheels for
Colab's Python/CUDA pair and takes tens of minutes to build from source, and
chumpy 0.70 imports on neither Python 3.11+ nor NumPy 2. The `compat/` package
covers both: pure-PyTorch rotation conversions, and a loader that reads the
official SMPL `.pkl` without chumpy. It also absorbs the defaults that moved in
newer releases, so the same code runs on both setups without version pins:

| Shim | Replaces |
| --- | --- |
| `compat.rotation_conversions` | `pytorch3d.transforms` |
| `compat.smpl.load_smpl` | `smplx.SMPL` on chumpy-backed model files |
| `compat.torch_load` | `torch.load`, whose `weights_only` default flipped in PyTorch 2.6 |
| `compat.estimate_tempo` | `librosa.beat.tempo`, moved in librosa 0.10 |
| `compat.matrix_sqrtm` | `scipy.linalg.sqrtm(disp=False)`, removed in SciPy 1.18 |

Check the setup with the test suite, which needs no dataset:

```bash
python -m unittest discover -s tests -t .
```

#### Known difference from the pinned environment

Training reads the music features shipped in `music.npy`, but inference
recomputes them from the audio with whatever librosa is installed, so the Colab
stack introduces a small train/inference mismatch that the pinned librosa 0.9.2
does not have. Measured on identical audio, 0.9.2 against 0.11.0:

| Feature block | Dims | Difference |
| --- | --- | --- |
| onset envelope | 1 | none (bit-identical) |
| MFCC | 20 | none (bit-identical) |
| chroma CENS | 12 | 2.1e-2 absolute, 2.5% relative (max) |
| onset peak one-hot | 1 | none |
| beat one-hot | 1 | 3 of 241 frames |

The chroma difference comes from `librosa.cqt`, whose `res_type` default moved
from `None` to `soxr_hq` in 0.10; the `auto_resample` branch that `None` used to
select was deleted, so the old behaviour cannot be restored through parameters.
The beat difference is internal to `beat_track` — the onset envelope feeding it
is identical and the starting BPM was held fixed. How much this shifts generated
motion has not been measured, and reproducing published numbers exactly still
calls for the pinned environment.

### Installation

The code was validated on Linux with Python 3.7.12, PyTorch 1.12.1, and CUDA
11.6. A CUDA GPU with at least 16 GB memory is recommended for training and
inference. PyTorch3D and chumpy are optional here too; the project uses them
when present and falls back to `compat/` otherwise.

1. Create the Conda environment.

```bash
conda create -n atomicdance python=3.7 -y
conda activate atomicdance
```

2. Install PyTorch and the base Python dependencies.

```bash
pip install -r requirements.txt
```

3. Install the packages that import or compile against PyTorch.

```bash
pip install git+https://github.com/rodrigo-castellon/jukemirlib.git@a91d87fcae0dd89085752421e794ea7e1b300735
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.1
```

If PyTorch3D fails to build, either install 0.7.1 separately with the matching
CUDA toolchain or skip it: `compat.rotation_conversions` provides the same
transforms in pure PyTorch and is used automatically when the import fails.

### Data Preparation

Download the processed atomic dataset from [Dataset](https://drive.google.com/file/d/1ETsaetMMWeKV3_E3Lr40BdybAsUAG8WM/view?usp=sharing) and extract it under
`data/atomic_aistpp/`. No additional label preprocessing is required.

```text
data/atomic_aistpp/
  manifest.json
  normalizer.pt
  train/
    motion.npy
    music.npy
    labels.npy
    names.json
  test/
    motion.npy
    music.npy
    labels.npy
    names.json
```

The released `atomic_aistpp` package is the only project-specific dataset that
needs to be downloaded. It contains the frame-aligned motion, 35-dimensional
music features, and atomic labels used for training and inference. Atomic labels
`1..100` represent movement categories; label `0` represents a transition.

Evaluation against AIST++ ground truth additionally expects motion PKLs and WAVs
under `data/edge_aistpp/{motions,wavs}`. Obtain AIST++ from its
[official website](https://google.github.io/aistplusplus_dataset/) rather than
from this project release. Feature extraction also requires the licensed SMPL
model at `smpl/SMPL_MALE.pkl`; obtain it from the
[official SMPL website](https://smpl.is.tue.mpg.de/).

## Training

### Atomic Movement Planner

```bash
python train_atomic.py \
  --stage planner \
  --data-root data/atomic_aistpp \
  --output-dir runs/atomic_planner \
  --device cuda \
  --epochs 20 \
  --batch-size 16
```

### Dance Completion Model


```bash
python train_atomic.py \
  --stage completion \
  --data-root data/atomic_aistpp \
  --output-dir runs/atomic_completion \
  --device cuda \
  --epochs 200 \
  --batch-size 8
```

Training reports mean loss every five epochs and saves a resumable checkpoint
every 20 epochs. Use `--resume CHECKPOINT` to continue training. Add
`--max-steps 10` for a bounded debugging run.

## Evaluation

The unified evaluator performs motion generation, feature extraction, caching,
and metric computation. It reports kinematic/manual-feature FID and diversity
and Beat Alignment Score (BAS). Prediction and ground-truth feature
distributions are standardized independently following the provided evaluation
starter.

The commands below evaluate the sequences in
`data/splits/crossmodal_test.txt`.

### Planner Plan + Dance Completion

This is the full two-stage inference setting. Atomic labels are generated by
the planner rather than read from ground truth.

```bash
python -m eval.evaluate \
  --ground-truth-motions data/edge_aistpp/motions \
  --audio-dir data/edge_aistpp/wavs \
  --sequence-list data/splits/crossmodal_test.txt \
  --plan-source planner \
  --planner-checkpoint runs/atomic_planner/<name>.pt \
  --completion-checkpoint runs/atomic_completion/<name>.pt \
  --atomic-data-root data/atomic_aistpp \
  --smpl-model smpl/SMPL_MALE.pkl \
  --device cuda:0 \
  --max-inference-frames 150 \
  --inference-batch-size 4 \
  --workers 4 \
  --inference-output eval/generated_planner \
  --cache-dir eval/cache_planner \
  --output eval/results_planner.json
```



Add `--overwrite-inference --force-extract` to regenerate motions and features
instead of reusing existing caches.

### Pretrained Checkpoints

Pretrained checkpoints will be released at [Checkpoints](https://drive.google.com/drive/folders/1r707t1FKhs_FkHNkNbqtDxIaXiYMUZuq?usp=sharing). The expected
layout is:

```text
runs/
  atomic_planner/
    planner_*.pt
  atomic_completion/
    completion_*.pt
```

## Citation

If you find this project useful, please consider citing our work. The entry
below will be updated when the final publication metadata is available.

```bibtex
@inproceedings{cai2026atomicdance,
  title={Music-to-Dance Generation via Atomic Movements},
  author={Cai, Xinhao and Sun, Yixuan and Zheng, Minghang and Chen, Qingchao and
          Jin, Xin and Zhu, Song-chun and Liu, Yang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

## Acknowledgements

This implementation is built on
[EDGE](https://github.com/Stanford-TML/EDGE). We also thank the authors of
[AIST++](https://google.github.io/aistplusplus_dataset/),
[PyTorch3D](https://github.com/facebookresearch/pytorch3d),
[SMPL](https://smpl.is.tue.mpg.de/), and the related music-to-dance generation
projects used in our experiments.

## License

This project is released under the license in [LICENSE](LICENSE). AIST++, SMPL,
pretrained models, and other third-party assets remain subject to their
respective licenses.
