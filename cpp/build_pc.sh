#!/usr/bin/env bash
# Configure and build the C++ core for Linux/WSL2 (docs/plan.md PC phase).
# Mirrors cpp/build.sh's role for cpp/pc/CMakeLists.txt: pins the PC conda
# prefix, builds into its own tree, and never touches the Mac build/CMake.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
BUILD_DIR="$REPO_ROOT/cpp/build_pc"
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--build-dir" || -z "$2" ]]; then
    printf 'Usage: %s [--build-dir PATH]\n' "$0" >&2
    exit 2
  fi
  if [[ "$2" = /* ]]; then
    BUILD_DIR="$2"
  else
    BUILD_DIR="$REPO_ROOT/$2"
  fi
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  printf 'cpp/build_pc.sh targets Linux or WSL2 only (uname -s reported %s)\n' "$(uname -s)" >&2
  exit 2
fi
PREFIX="${CATJET_CPP_PREFIX:-$HOME/miniforge3/envs/catjet-cpp-pc}"
if [[ ! -x "$PREFIX/bin/cmake" ]]; then
  printf 'No cmake at %s/bin/cmake; create the catjet-cpp-pc conda env first (see PC_SETUP.md)\n' "$PREFIX" >&2
  exit 2
fi
"$PREFIX/bin/cmake" -S "$HERE/pc" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PREFIX/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCANTERA_PREFIX="$PREFIX" \
  -DPython_EXECUTABLE="$REPO_ROOT/.venv/bin/python" \
  -Dpybind11_DIR="$("$REPO_ROOT/.venv/bin/python" -m pybind11 --cmakedir)"
"$PREFIX/bin/cmake" --build "$BUILD_DIR"
