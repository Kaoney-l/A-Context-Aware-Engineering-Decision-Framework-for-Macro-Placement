# CarePlace: A Context-Aware Engineering Decision Framework for Chip Physical Design Automation

Official implementation of **CarePlace**, a context-aware reward framework for chip macro placement. CarePlace couples macro-level perception, preference inference, and reward feedback in a unified decision process.

## Requirements
+ python==3.8.5
+ torch==1.7.1
+ torchvision==0.8.2
+ torchaudio==0.7.2
+ pyyaml==5.3.1
+ gym==0.22.0
+ Shapely==2.0.4
+ matplotlib==3.4.3
+ cairocffi==1.7.0
+ tqdm==4.61.2
+ tensorboard==2.14.0
+ scikit_learn==1.3.2
+ numpy==1.21.2

## File structure

+ `benchmark/` — ICCAD 2015 benchmark suite (download separately).
+ `config/` — Hyperparameters including CarePlace-specific settings.
+ `DREAMPlace_source/` — Third-party standard-cell placer from [DREAMPlace](https://github.com/limbo018/DREAMPlace).
+ `policy/` — Pretrained model checkpoints.
+ `src/` — Core source code:
  + `main.py` — Training and evaluation entry point.
  + `agent.py` — PPO agent with integrated PreferencePredictor.
  + `place_env/place_env.py` — Gym environment with structural mask, dynamic group calibration, and perception-guided feedback.
  + `problem_instance.py` — Benchmark loading and macro interaction graph construction.
  + `model/actor.py` — Policy network with preference-aware mask weighting.
  + `model/critic.py` — Value network.
  + `model/cnn.py` — CNN backbones (MyCNN, MyCNNCoarse with ResNet18).
  + `model/preference_predictor.py` — GNN encoder, topological interaction descriptor, spatial encoder, preference prediction head, and dynamic group calibration.
+ `utils/` — State parsing, HPWL/regularity computation, LEF/DEF parsing.

## CarePlace Hyperparameters

Key settings in `config/default.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `use_perception_guidance` | `True` | Enable preference-guided feedback and mask weighting |
| `use_reward_scaling` | `False` | Legacy min-max scaling (replaced by dynamic group calibration) |
| `wire_coeff` | `0.7` | Initial static trade-off coefficient β₀ |
| `preference_anneal_start` | `0.0` | Initial λ for preference annealing |
| `preference_anneal_end` | `1.0` | Final λ for preference annealing |
| `preference_anneal_episodes` | `500` | Episodes to anneal λ over |
| `preference_gnn_layers` | `2` | GNN message-passing depth |
| `preference_gnn_out_dim` | `32` | GNN embedding dimension |
| `preference_topo_out_dim` | `32` | Topological interaction descriptor output dim |
| `preference_spatial_dim` | `64` | Spatial encoder output dimension |
| `calibration_epsilon` | `1e-8` | Numerical stability for dynamic calibration |

## Usage

### Setup
The base environment requires DREAMPlace. You can either use the pre-built Docker image or build DREAMPlace directly from source.

**Build DREAMPlace from source:** Follow the instructions at [DREAMPlace](https://github.com/limbo018/DREAMPlace) to set up the environment, then compile `DREAMPlace_source/` as above.

Download the ICCAD 2015 benchmark from [Google Drive](https://drive.google.com/file/d/1JEC17FmL2cM8BEAewENvRyG6aWxH53mX/view?usp=sharing) and place it in `benchmark/`.

### Parameters
+ `--seed` — Random seed.
+ `--gpu` — GPU ID.
+ `--episode` — Number of training episodes.
+ `--checkpoint_path` — Path to saved model for loading.
+ `--eval_policy` — Evaluation-only mode (requires `--checkpoint_path`).
+ `--dataset_path` — Optional placement file to regulate. If not provided, CarePlace initializes from DREAMPlace.

### Training
```bash
cd src
python main.py --benchmark_train=[superblue1] --benchmark_eval=[superblue1]
```
Script `run_train.sh` is provided for quick start.

### Testing
```bash
python main.py --benchmark_train=[] --benchmark_eval=[superblue1] \
               --check_point_path=../policy/pretrained_model.pkl --eval_policy=True
```
Script `run_test.sh` is provided for quick start.

