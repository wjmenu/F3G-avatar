# AvatarReX ZZR configs

Default entry points (at this directory root):
- **avatar.yaml** – main training config
- **template.yaml** – template training

Grouped configs (by experiment type):

| Subdir | Description |
|--------|-------------|
| **face_pretrain/** | Face pretraining (then train): 100/200 test, 1k steps |
| **100k_smplxpp/** | 100k body training (SMPL-X++), face variants, +40k |
| **100k_animatable_gs/** | 100k animatable gaussians run |
| **10k/** | 10k runs: smplxpp, face_gcn6, animatable_gs |
| **ablations/** | Ablations A/B/C: smplx vs old template vs smplxpp template |
| **baselines/** | Baseline configs: SMPL-X, old template |
| **pt1000_tr2000/** | Short runs: pretrain 1k, train 2k (A/B/C) |
| **pt5k_tr100k/** | Long runs: pretrain 5k, train 100k (smplx, old template, smplxpp template) |

Use with `-c configs/avatarrex_zzr/<subdir>/<config>.yaml` (or `-c configs/avatarrex_zzr/avatar.yaml` for default).
