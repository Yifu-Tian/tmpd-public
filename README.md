# Topological Motion Planning Diffusion: Generative Tangle-Free Path Planning for Tethered Robots in Obstacle-Rich Environments


Tian, Y.; Xu, X.; Nguyen, T.-M.; Cao, M. (2026). **_Topological Motion Planning Diffusion: Generative Tangle-Free Path Planning for Tethered Robots in Obstacle-Rich Environments_**, submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

[<img src="https://img.shields.io/badge/arXiv-2603.26696-b31b1b.svg?&style=for-the-badge&logo=arxiv&logoColor=white" />](https://arxiv.org/pdf/2603.26696)


[TODO: add TMPD paper figures/GIFs here]

---
This repository implements `TMPD` - Topological Motion Planning Diffusion -, a method for tethered robots path planning with diffusion models, submitted to IROS 2026.

**NOTES**

This codebase is developed based on the [mpd-public](https://github.com/joaoamcarvalho/mpd-public/tree/main).

If you have any questions please let me know -- [yifutian@link.cuhk.edu.cn](mailto:yifutian@link.cuhk.edu.cn)

---
## Installation

Pre-requisites:
- Ubuntu 20.04
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)

Clone this repository with
```bash
cd ~
git clone --recurse-submodules https://github.com/Yifu-Tian/tmpd-public.git
cd tmpd-public
```

Download [IsaacGym Preview 4](https://developer.nvidia.com/isaac-gym) and extract it under `deps/isaacgym`
```bash
mv ~/Downloads/IsaacGym_Preview_4_Package.tar.gz ~/tmpd-public/deps/
cd ~/tmpd-public/deps
tar -xvf IsaacGym_Preview_4_Package.tar.gz
```

Run the setup script to install dependencies.
```bash
cd ~/tmpd-public
bash setup.sh
```

---
## Running the TMPD inference

To run TMPD / baseline inference, first download trajectories and trained models.

```bash
conda activate mpd
```

```bash
gdown --id 1mmJAFg6M2I1OozZcyueKp_AP0HHkCq2k
tar -xvf data_trajectories.tar.gz
gdown --id 1I66PJ5QudCqIZ2Xy4P8e-iRBA8-e2zO1
tar -xvf data_trained_models.tar.gz
```

Run benchmark scripts
```bash
cd scripts/inference
python run_astar.py
python run_rrt.py
python run_mpd.py
python run_tmpd.py
```

Or run the full pipeline
```bash
cd scripts/inference
bash run_all.sh
```

Optional plotting
```bash
cd scripts/inference
python plot_all_figures.py
```

Result folders are generated under `scripts/inference/` and/or `data_trained_models/[model_id]/results_inference/` depending on script settings.

---
## Generate data and train from scratch

We recommend running the following in a SLURM cluster.

```bash
conda activate mpd
```

To regenerate the data:
```bash
cd scripts/generate_data
python launch_generate_trajectories.py
```

To train the model:
```bash
cd scripts/train_diffusion
python launch_train_01.py
```

---
## Citation

If you use our work or code base(s), please cite:
```latex
@article{tian2026tmpd,
  title={Topological Motion Planning Diffusion: Generative Tangle-Free Path Planning for Tethered Robots in Obstacle-Rich Environments},
  author={Tian, Yifu and Xu, Xinhang and Nguyen, Thien-Minh and Cao, Muqing},
  journal={arXiv preprint arXiv:2603.26696},
  year={2026},
  note={Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  url={https://arxiv.org/pdf/2603.26696}
}
```

If you also build on MPD, please additionally cite:
```latex
@inproceedings{carvalho2023mpd,
  title={Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models},
  author={Carvalho, J. and Le, A.T. and Baierl, M. and Koert, D. and Peters, J.},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```
