#!/usr/bin/env bash
# 下载 onnxruntime CPU 官方发行包到 lib/<version>/，
# 并下载 yolo11n.onnx 模型与测试图片，供跨版本 / 跨平台测试矩阵使用。
#
# 用法:
#   scripts/fetch_deps.sh                  # 全部 5 个版本 (1.23.2 ~ 1.27.1)，自动识别平台
#   scripts/fetch_deps.sh 1.23.2 1.27.1    # 指定版本
#
# 平台: Linux x64 / macOS universal2 / Windows (Git Bash / WSL，用 unzip 或 tar 解 zip)
set -euo pipefail

VERSIONS=("$@")
if [ ${#VERSIONS[@]} -eq 0 ]; then
    VERSIONS=("1.23.2" "1.24.4" "1.25.1" "1.26.0" "1.27.1" "1.28.3" "1.29.1" "1.30.1" "1.31.1")
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/lib"
mkdir -p "$DEST"

# 识别平台资产名
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Linux)
        case "$ARCH" in
            aarch64|arm64) ASSET_OS="linux-aarch64" ;;
            *) ASSET_OS="linux-x64" ;;
        esac
        EXT="tgz" ;;
    Darwin)
        case "$ARCH" in
            arm64) ASSET_OS="osx-arm64" ;;
            *) ASSET_OS="osx-x86_64" ;;
        esac
        EXT="tgz" ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        case "${PROCESSOR_ARCHITECTURE:-}" in
            ARM64) ASSET_OS="win-arm64" ;;
            *) ASSET_OS="win-x64" ;;
        esac
        EXT="zip" ;;
    *)
        echo "unsupported OS: $OS"; exit 1 ;;
esac

for v in "${VERSIONS[@]}"; do
    dir="$DEST/$v"
    if [ -f "$dir/onnxruntime.dll" ] || [ -f "$dir/libonnxruntime.so" ] || [ -f "$dir/libonnxruntime.dylib" ]; then
        echo "onnxruntime $v already present, skip"
        continue
    fi
    archive="/tmp/onnxruntime-${ASSET_OS}-$v.$EXT"
    url="https://github.com/microsoft/onnxruntime/releases/download/v$v/onnxruntime-${ASSET_OS}-$v.$EXT"
    echo "downloading $url"
    curl -sSL --retry 3 -o "$archive" "$url"
    mkdir -p "$dir"
    tmpdir="$(mktemp -d)"
    # tar 在 Git Bash / macOS / Linux 下均可解 zip 与 tgz
    tar -xf "$archive" -C "$tmpdir"
    inner="onnxruntime-${ASSET_OS}-$v"
    if [ -d "$tmpdir/$inner" ]; then
        cp "$tmpdir/$inner/lib/"* "$dir/"
    else
        cp "$tmpdir"/lib/* "$dir/"
    fi
    rm -rf "$tmpdir" "$archive"
done

model_dir="$ROOT/examples/yolo/testdata"
mkdir -p "$model_dir"
if [ ! -f "$model_dir/yolo11n.onnx" ]; then
    echo "downloading yolo11n.onnx"
    curl -sSL --retry 3 -o "$model_dir/yolo11n.onnx" \
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
fi
if [ ! -f "$model_dir/bus.jpg" ] && [ ! -f "$model_dir/yolo_demo.png" ] && [ ! -f "$model_dir/yolo_demo.jpg" ]; then
    echo "downloading bus.jpg"
    curl -sSL --retry 3 -o "$model_dir/bus.jpg" "https://ultralytics.com/images/bus.jpg"
fi

echo "done."
