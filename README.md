## F3G-Avatar : Face Focussed Full-body Avatar
### CVPRW 2026 | [Project Page](https://wjmenu.github.io/F3G-avatar/) | [Paper](https://arxiv.org/abs/2604.09835) | [🤗 Hugging Face](https://huggingface.co/wjmenu/F3G-avatar)

![F3G-Avatar pipeline overview](assets/pipeline.png)

Official implementation of **F3G-Avatar**, our method for building animatable full-body human avatars from calibrated multi-view RGB captures.

# Installation

## 1. Environment

```bash
git clone https://github.com/wjmenu/F3G-avatar.git
cd F3G-avatar

conda create -n animatable_gaussians python=3.10 -y
conda activate animatable_gaussians
pip install -r requirements.txt
```
 
## 2. Build diff-gaussian rasterization and StyleAvatar

These CUDA extensions are required for **training and inference** (`AvatarNet` uses DualStyleUNet from [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar)).

```bash
# diff-gaussian-rasterization (depth + alpha)
cd gaussians/diff_gaussian_rasterization_depth_alpha
python setup.py install
cd ../..

# StyleUNet (from StyleAvatar; pose-conditioned color / geometry networks)
cd network/styleunet
python setup.py install
cd ../..
```


## 3. SMPL-X body models

Download [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php) and place the model files under `./smpl_files/smplx/`.

## 4. NeuS2 (mesh reconstruction)

[NeuS2](https://github.com/19reborn/NeuS2) reconstructs a clothed surface mesh from multiview images. It is used when running `pipeline_multiview_to_template.py` with `--run_neus2` (Data Preparation, step 2).

```bash
# If missing:
git clone --recursive https://github.com/19reborn/NeuS2.git othercode/NeuS2

cd othercode/NeuS2
cmake . -B build
cmake --build build --config RelWithDebInfo -j
cd ../..
```


## 5. MHR template pipeline ([4D-Dress](https://github.com/eth-ait/4d-dress) + [PhysAvatar](https://github.com/y-zheng18/PhysAvatar) + [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar))

**4D-Dress** and **PhysAvatar** work together in `pipeline_multiview_to_template.py`: 4D-Dress labels garment regions on the reconstructed mesh; PhysAvatar composes the MHR template and LBS weights. Both live under `othercode/` (clone if your checkout does not include them).

**4D-Dress** — parsing for semantic split and face crops:

```bash
git clone https://github.com/eth-ait/4d-dress.git othercode/4d-dress
```

Download checkpoints ([4D-Dress model install](https://github.com/eth-ait/4d-dress#model-installation)):

- **Graphonomy** (required): `inference.pth` → `othercode/4d-dress/4dhumanparsing/checkpoints/graphonomy/`
- **SAM** (optional, face crops): `sam_vit_h_4b8939.pth` → `othercode/4d-dress/4dhumanparsing/checkpoints/sam/`

**PhysAvatar** — template composition and weight inpainting:

```bash
git clone https://github.com/y-zheng18/PhysAvatar.git othercode/PhysAvatar
```


# Data Preparation

Training needs more than raw images: a **clothed body template (MHR)** with pose maps for the full body, and **head crops** for the face branch. Below is the usual path from a multiview capture to those assets.

Set `export DATA=/path/to/your_subject` and run commands from the repo root.

## Raw capture

Your subject folder should contain:

- `calibration_full.json` — camera intrinsics and extrinsics per view  
- `smpl_params.npz` — SMPL-X pose sequence  
- One folder per camera (`0`, `1`, … or AvatarReX-style IDs like `22010708`)  
- Per frame: `<cam>/%08d.jpg` and `<cam>/mask/pha/%08d.jpg`

## Face crops

The face network trains on tight head views, not full-frame images.  
`tools/python/crop_dataset_rex_faces.py` detects the head (4D-DRESS / Graphonomy, optional SAM), crops each frame, updates calibration, and writes a separate dataset:

```bash
python tools/python/crop_dataset_rex_faces.py --src_root "$DATA" --dst_root /path/to/crops
```

Point `train.data.crop_data_dir` in your YAML to `/path/to/crops` (can differ from `data_dir`). Use `--max_frames N` or `--no_sam` while testing.

## MHR body, position maps, training, and evaluation

The body uses **MHR** (mesh human representation): a T-pose mesh that includes outer clothing, plus per-vertex skinning weights.

**1. Package multiview input** (`mesh_data`):

```bash
python create_template.py --data_dir "$DATA" --out_dir mesh_data
cp mesh_data/calibration_full.json mesh_data/transforms.json
```

Masked RGBA images per camera, for NeuS2 / the mesh pipeline.

**2. Build template** (`mesh`) — full outfit (shirt + pants): keep `--outfit Outer`:

```bash
python tools/python/pipeline_multiview_to_template.py \
  --images_dir mesh_data/images --transforms_json mesh_data/transforms.json \
  --smpl_params_npz mesh_data/smpl_params.npz --out_dir mesh \
  --run_neus2 --neus2_name neus2_exp --neus2_steps 100000 --outfit Outer --weights_mode inpaint
```

Outputs: `mesh/semantic_split/...`, MHR template at `mesh/template/tpose/template.obj` and `mesh/template/weights/template_weights.npy`. Intermediate compose files go to `mesh/_compose_work/`.
If you already have a mesh: add `--mesh_obj /path/to/mesh.obj` and drop the NeuS2 flags.

**3. Rasterize pose maps** into the training dataset:

Copy and edit a config (e.g. `configs/avatarrex_zzr/avatar.yaml`) so `train.data.data_dir` points to `$DATA`, then run:

```bash
python -m gen_data.gen_pos_maps -c configs/avatarrex_zzr/avatar.yaml
```

Requires `mesh/template/` from step 2 (auto-detected when run from repo root). Writes `$DATA/smpl_pos_map/` (`cano_smpl_pos_map.exr`, `init_pts_lbs.npy`, and per-frame `%08d.exr`).

## Training

Set paths in your YAML under `train`:
- `data.data_dir` — full-body dataset (`$DATA`)
- `data.crop_data_dir` — face-crop dataset from [Face crops](#face-crops) (optional; same as `data_dir` if unused)
- `net_ckpt_dir` — where checkpoints and eval images are saved

```bash
python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml -m train
```

## Evaluation

Set `test.prev_ckpt` to a trained checkpoint (e.g. `./results/avatarrex_zzr/avatar/epoch_latest`) and configure `test.data` or `test.pose_data` for the frames to render:

```bash
python main_avatar.py -c configs/avatarrex_zzr/avatar.yaml -m test
```

Outputs go to `test.output_dir`, or by default `./test_results/<subject>/<experiment>/`. Useful `test` options: `render_view_idx`, `view_setting` (`free` or `camera`), `save_ply`, `save_tex_map`, `n_pca` / `sigma_pca` for pose variation.

# TODOS
- [x] Release the code.
- [ ] Simplify the pre-processing 

# Acknowledgement

We build on the following projects (setup: [Installation](#installation) §4–5):

- [Animatable Gaussians](https://github.com/lizhe00/AnimatableGaussians) — base animatable 3D Gaussian avatar framework
- [NeuS2](https://github.com/19reborn/NeuS2) — fast neural surface reconstruction for clothed mesh capture
- [4D-Dress](https://github.com/eth-ait/4d-dress) + [PhysAvatar](https://github.com/y-zheng18/PhysAvatar) + [StyleAvatar](https://github.com/LizhenWangT/StyleAvatar) — garment parsing, MHR template / LBS composition, and DualStyleUNet avatar network
- [3D Gaussian Splatting](https://github.com/ashawkey/diff-gaussian-rasterization) — differentiable Gaussian rasterization

# Citation

If you find our code or data helpful to your research, please consider citing our paper:

```bibtex
@misc{menu2026f3gavatarfacefocused,
  title={F3G-Avatar : Face Focused Full-body Gaussian Avatar},
  author={Willem Menu and Erkut Akdag and Pedro Quesado and Yasaman Kashefbahrami and Egor Bondarev},
  year={2026},
  eprint={2604.09835},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2604.09835},
}
```

