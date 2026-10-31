#!/usr/bin/env bash
# Build oxDNA (with CUDA) inside WSL Ubuntu and install it to /usr/local/bin/oxDNA.
# Run from Windows:   wsl -d Ubuntu -- bash /mnt/c/synbio-ai/scripts/setup_oxdna_wsl.sh
# CPU-only build:     WITH_CUDA=0 bash setup_oxdna_wsl.sh   (then set backend: CPU in configs/oxdna.yaml)
# WSL2 uses the Windows NVIDIA driver; do NOT install a Linux GPU driver, only the CUDA toolkit.
set -euo pipefail
WITH_CUDA="${WITH_CUDA:-1}"
SRC="${OXDNA_SRC:-$HOME/oxDNA}"

sudo apt-get update
sudo apt-get install -y build-essential cmake git g++-12

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
if [[ "$WITH_CUDA" == "1" ]]; then
  sudo apt-get install -y nvidia-cuda-toolkit
  # Ubuntu's CUDA toolkit rejects newer GCC versions as host compiler; pin g++-12.
  CMAKE_ARGS+=(-DCUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12)
fi

[[ -d "$SRC" ]] || git clone --depth 1 https://github.com/lorenzo-rovigatti/oxDNA.git "$SRC"
mkdir -p "$SRC/build" && cd "$SRC/build"
cmake .. "${CMAKE_ARGS[@]}"
make -j"$(nproc)" oxDNA
sudo install -m 755 bin/oxDNA /usr/local/bin/oxDNA
echo "installed: $(command -v oxDNA)"
if [[ "$WITH_CUDA" == "1" ]]; then nvidia-smi --query-gpu=name --format=csv,noheader || echo "WARNING: GPU not visible in WSL"; fi
