# Hugging Face setup for F3G-Avatar

Efficient layout (recommended):

| Platform | Role |
|----------|------|
| [GitHub](https://github.com/wjmenu/F3G-avatar) | Source code, issues, docs, project page |
| [Hugging Face](https://huggingface.co/wjmenu/F3G-avatar) | Model card, figures, pretrained checkpoints |

This mirrors [erkutt/MTFL_UCF_Crime](https://huggingface.co/erkutt/MTFL_UCF_Crime) **without duplicating the full codebase** on HF (MTFL uploads code; we link to GitHub instead — easier to maintain).

## One-time setup

```bash
pip install -U huggingface_hub
hf auth login   # use a token with **Write** access (not read-only)
```

## Create the repo (web UI)

1. https://huggingface.co/new
2. **Owner:** `wjmenu`
3. **Name:** `F3G-Avatar`
4. **Type:** Model
5. **License:** Apache 2.0 (or your choice)
6. Create (empty repo)

## Publish the model card + figures

From this folder (`huggingface/`):

```bash
cd huggingface

# Copy figures from the GitHub repo / project page
mkdir -p figures
cp ../assets/pipeline.png figures/
cp ../assets/result.jpeg figures/

# Clone empty HF repo (skip git lfs if not installed on cluster)
git clone https://huggingface.co/wjmenu/F3G-avatar hf-repo
cd hf-repo

# Copy card + figures
cp ../README.md ../.gitattributes .
mkdir -p checkpoints figures
cp ../figures/* figures/
touch checkpoints/.gitkeep

git add .
git commit -m "Add F3G-Avatar model card and figures"
git push
```

**Or upload without git** (works without git-lfs; needs write token):

```bash
cd /home/wmenu/F3G-avatar/huggingface/hf-repo
hf upload wjmenu/F3G-avatar . . --commit-message "Add F3G-Avatar model card and figures"
```

## Upload checkpoints later

```bash
hf upload wjmenu/F3G-avatar \
  /path/to/epoch_latest.pt \
  checkpoints/avatarrex_zzr/epoch_latest.pt
```

Or copy files into the cloned repo under `checkpoints/` and `git push`.

## Link from GitHub README

Add near the top of `README.md`:

```markdown
🤗 [Pretrained models on Hugging Face](https://huggingface.co/wjmenu/F3G-avatar)
```

## Optional: HF Space (interactive demo)

Later, create a **Space** (Gradio) at https://huggingface.co/new-space and point it at inference code + a checkpoint from this model repo.
