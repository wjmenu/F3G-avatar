#!/usr/bin/env bash
# Create a clean release branch with only template-pipeline + avatar-training code.
# Usage: bash scripts/create_f3g_release_branch.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BRANCH="${1:-release-candidate}"

echo "[1/4] Creating orphan branch: ${BRANCH}"
git branch -D "${BRANCH}" 2>/dev/null || true
git checkout --orphan "${BRANCH}"

echo "[2/4] Clearing index"
git rm -rf --cached . >/dev/null 2>&1 || true

echo "[3/4] Staging release files"
RELEASE_PATHS=(
  README.md
  requirements.txt
  .gitignore
  config.py
  main_avatar.py
  create_template.py
  assets/pipeline.png
  assets/result.png
  index.html
  style.css
  nojekyll.txt
  othercode/README.md
  smpl_files/README.md
  smpl_files/mano
  network
  gaussians
  dataset
  gen_data
  utils
  smplx
  configs/avatarrex_zzr
  tools/__init__.py
  tools/python/pipeline_multiview_to_template.py
  tools/python/compose_smplxpp_physavatar.py
  tools/python/mesh_semantic_separation_4ddress.py
  tools/python/convert_dataset_rex_smpl_params_to_physavatar_pth.py
  tools/python/crop_dataset_rex_faces.py
  tools/python/export_smplxpp_template_for_anigs.py
)

for path in "${RELEASE_PATHS[@]}"; do
  if [[ -e "${path}" ]]; then
    git add "${path}"
    echo "  + ${path}"
  else
    echo "  ! missing (skipped): ${path}" >&2
  fi
done

echo "[4/4] Committing"
git status --short | wc -l | xargs -I{} echo "  {} paths staged"
git commit -m "$(cat <<'EOF'
Release: MHR template pipeline and F3G-Avatar training code.

Includes data preparation (create_template, pipeline_multiview_to_template),
position-map generation, and main_avatar training. External deps (NeuS2,
4D-Dress, PhysAvatar) and SMPL-X models are downloaded separately; see README.
EOF
)"

echo ""
echo "Done. Branch '${BRANCH}' is ready."
echo "Review locally, then push privately:"
echo "  git push f3g ${BRANCH}:${BRANCH}"
echo ""
echo "After review, publish to main (rewrites remote history):"
echo "  git push f3g ${BRANCH}:main --force"
echo ""
echo "Return to previous work:"
echo "  git checkout master"
