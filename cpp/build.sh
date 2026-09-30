#!/usr/bin/env bash
# Configure and build the C++ core. Apple clang from the Command Line Tools with
# the MacOSX26.5 SDK: xcode-select points to a broken Xcode.app and the default
# MacOSX27.0 SDK cannot be linked here (registration amendment P8-A1.3).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export DEVELOPER_DIR=/Library/Developer/CommandLineTools
PREFIX="${CATJET_CPP_PREFIX:-$HOME/miniforge3/envs/catjet-cpp}"
SDK="${CATJET_SDK:-$DEVELOPER_DIR/SDKs/MacOSX26.5.sdk}"
"$PREFIX/bin/cmake" -S "$HERE" -B "$HERE/build" -G Ninja \
  -DCMAKE_MAKE_PROGRAM="$PREFIX/bin/ninja" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$DEVELOPER_DIR/usr/bin/clang++" \
  -DCMAKE_OSX_SYSROOT="$SDK" \
  -DCANTERA_PREFIX="$PREFIX" \
  -DPython_EXECUTABLE="$HERE/../.venv/bin/python" \
  -Dpybind11_DIR="$("$HERE/../.venv/bin/python" -m pybind11 --cmakedir)"
"$PREFIX/bin/cmake" --build "$HERE/build"
