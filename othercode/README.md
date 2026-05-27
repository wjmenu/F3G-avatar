# External dependencies (not shipped in this repo)

Clone these into this folder before running the MHR template pipeline. See [README.md](../README.md) for setup commands.

| Folder | Purpose | Clone |
|--------|---------|--------|
| `NeuS2/` | Clothed mesh reconstruction | `git clone --recursive https://github.com/19reborn/NeuS2.git NeuS2` |
| `4d-dress/` | Garment parsing (Graphonomy) | `git clone https://github.com/eth-ait/4d-dress.git 4d-dress` |
| `PhysAvatar/` | MHR template composition + LBS | `git clone https://github.com/y-zheng18/PhysAvatar.git PhysAvatar` |

Download 4D-Dress checkpoints into `4d-dress/4dhumanparsing/checkpoints/` (Graphonomy required; SAM optional for face crops).

Build NeuS2 after cloning (`cmake . -B build && cmake --build build --config RelWithDebInfo -j`).
