# TMPD Public

Topology-aware Motion Planning Diffusion (`TMPD`) built on top of the original MPD codebase.

This repository focuses on **lifelong / multi-segment navigation** in dynamic 2D obstacle fields, and compares:

- `Topo-A*`
- `Topo-RRT`
- `Vanilla MPD`
- `TMPD (Ours)`

The core idea is to augment diffusion-based planning with **global topological memory** (winding-signature based), so each new segment is planned with awareness of previously executed path topology.

---

## 1. What This Project Does

Compared to the original MPD release, this repo adds:

- Dynamic environment variants with obstacle dropout + newly inserted obstacles.
- Topology utility functions:
  - trajectory signature (`winding number` style)
  - topological energy evaluation
  - taut homotopy reference extraction
  - safety checks for sphere/box obstacles
- TMPD inference and benchmarking scripts for fair side-by-side comparison with classic topology-aware planners and vanilla MPD.

Main implementation files:

- `mpd/utils/topology_utils.py`
- `tmpd_baselines/environment/env_dense_2d_extra_objects.py`
- `tmpd_baselines/environment/env_simple_2d_extra_objects.py`
- `scripts/inference/run_astar.py`
- `scripts/inference/run_rrt.py`
- `scripts/inference/run_mpd.py`
- `scripts/inference/run_tmpd.py`
- `scripts/inference/plot_all_figures.py`

---

## 2. Environment Setup

Recommended:

- Ubuntu 20.04+
- CUDA GPU (for MPD / TMPD runs)
- Conda

Clone with submodules:

```bash
git clone --recurse-submodules https://github.com/Yifu-Tian/tmpd-public.git
cd tmpd-public
```

Download and extract Isaac Gym Preview 4 under `deps/isaacgym`:

```bash
mv ~/Downloads/IsaacGym_Preview_4_Package.tar.gz ./deps/
cd deps
tar -xvf IsaacGym_Preview_4_Package.tar.gz
cd ..
```

Install dependencies:

```bash
bash setup.sh
```

`setup.sh` creates the `mpd` conda environment and installs editable local dependencies:

- `deps/experiment_launcher`
- `deps/torch_robotics`
- `deps/motion_planning_baselines`
- `deps/isaacgym/python`
- `deps/storm`
- this repository (`pip install -e .`)

---

## 3. Prepare Data / Trained Models

The repository intentionally does **not** track heavy artifacts (datasets, checkpoints, benchmark image dumps, pickle logs).

If you want to run inference directly, download the pretrained assets:

```bash
conda activate mpd
gdown --id 1mmJAFg6M2I1OozZcyueKp_AP0HHkCq2k
tar -xvf data_trajectories.tar.gz
gdown --id 1I66PJ5QudCqIZ2Xy4P8e-iRBA8-e2zO1
tar -xvf data_trained_models.tar.gz
```

Expected local directories after extraction:

- `data_trajectories/`
- `data_trained_models/`

---

## 4. Run Benchmarks

All commands below are run from:

```bash
cd scripts/inference
```

Run each method separately:

```bash
python run_astar.py
python run_rrt.py
python run_mpd.py
python run_tmpd.py
```

One-click execution (all methods + final comparison plotting):

```bash
bash run_all.sh
```

Generate 1x4 comparison figures from shared trial data:

```bash
python plot_all_figures.py
```

---

## 5. Data Generation and Training

Generate training trajectories:

```bash
cd scripts/generate_data
python launch_generate_trajectories.py
```

Train diffusion models:

```bash
cd scripts/train_diffusion
python launch_train_01.py
```

---

## 6. Repository Layout

```text
mpd/                          # Core MPD modules (datasets, model, trainer, utils)
tmpd_baselines/               # Dynamic environment variants used by TMPD benchmarks
scripts/generate_data/        # Trajectory generation pipeline
scripts/train_diffusion/      # Diffusion model training pipeline
scripts/inference/            # TMPD + baselines inference/benchmark scripts
deps/                         # Git submodules (torch_robotics, baselines, etc.)
figures/                      # Lightweight demo GIFs
```

---

## 7. Notes on Open-Source Snapshot

- Large local files and generated benchmark outputs are excluded from Git.
- This keeps the repo lightweight and reproducible as source code.
- If you need exact experiment outputs, regenerate them from scripts in `scripts/inference/`.

---

## 8. Citation

If you use the original MPD method, please cite:

```bibtex
@inproceedings{carvalho2023mpd,
  title={Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models},
  author={Carvalho, J. and Le, A.T. and Baierl, M. and Koert, D. and Peters, J.},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```

---

## 9. Acknowledgements

- Original MPD repository and paper:
  - https://github.com/jacarvalho/mpd-public
  - https://arxiv.org/abs/2308.01557
- Diffuser:
  - https://github.com/jannerm/diffuser
