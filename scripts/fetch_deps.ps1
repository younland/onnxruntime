param(
    [string[]]$Versions = @("1.23.2", "1.24.4", "1.25.1", "1.26.0", "1.27.1", "1.28.3", "1.29.1", "1.30.1", "1.31.1"),
    [string]$Dest = "$PSScriptRoot\..\lib"
)

# 下载 onnxruntime CPU 官方发行包（Windows x64 / arm64 自动识别）到 lib/<version>/（默认 1.23 ~ 1.31 全部 9 个版本），
# 并下载 yolo11n.onnx 模型与 bus.jpg 测试图片，供跨版本测试矩阵使用。
# CUDA / TensorRT 构建请自行从 NVIDIA 渠道获取，本脚本只拉 CPU 版。

$ErrorActionPreference = "Stop"
$Dest = Join-Path (Resolve-Path "$PSScriptRoot\..").Path "lib"
$AssetArch = if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') { 'win-arm64' } else { 'win-x64' }

foreach ($v in $Versions) {
    $dir = Join-Path $Dest $v
    if (Test-Path (Join-Path $dir "onnxruntime.dll")) {
        Write-Host "onnxruntime $v already present, skip"
        continue
    }
    $zip = "$env:TEMP\onnxruntime-$AssetArch-$v.zip"
    $url = "https://github.com/microsoft/onnxruntime/releases/download/v$v/onnxruntime-$AssetArch-$v.zip"
    Write-Host "downloading $url"
    Invoke-WebRequest -Uri $url -OutFile $zip -TimeoutSec 600
    Expand-Archive -Path $zip -DestinationPath "$env:TEMP\ort-$v" -Force
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item "$env:TEMP\ort-$v\onnxruntime-$AssetArch-$v\lib\*" $dir
    Remove-Item $zip -Force
}

$modelDir = Join-Path (Resolve-Path "$PSScriptRoot\..").Path "examples\yolo\testdata"
New-Item -ItemType Directory -Force $modelDir | Out-Null
$model = Join-Path $modelDir "yolo11n.onnx"
if (-not (Test-Path $model)) {
    Write-Host "downloading yolo11n.onnx"
    Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx" -OutFile $model -TimeoutSec 600
}
$image = Join-Path $modelDir "bus.jpg"
if (-not (Test-Path $image) -and -not (Test-Path (Join-Path $modelDir "yolo_demo.png")) -and -not (Test-Path (Join-Path $modelDir "yolo_demo.jpg"))) {
    Write-Host "downloading bus.jpg"
    Invoke-WebRequest -Uri "https://ultralytics.com/images/bus.jpg" -OutFile $image -TimeoutSec 300
}

Write-Host "done. lib layout:"
Get-ChildItem $Dest -Recurse -Filter "onnxruntime.dll" | ForEach-Object { $_.FullName }
