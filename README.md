<div align="center">

<h1>onnxruntime (purego)</h1>

<p><strong>用 purego 直接调用 onnxruntime C API 的 Go 绑定：纯 Go、无 cgo、运行时动态加载 onnxruntime 动态库</strong></p>

<p>无 CGO 交叉编译 · 单二进制适配 ORT 1.23 ~ 1.31 · CPU 全版本 · CUDA/TensorRT 条件启用 · Windows / Linux / macOS（x64 + arm64）</p>

<p>
  <a href="go.mod"><img src="https://img.shields.io/badge/Go-1.24%2B-00ADD8?style=for-the-badge&logo=go&labelColor=gray" alt="Go"></a>
  <a href="go.mod"><img src="https://img.shields.io/badge/purego-v0.11.1-8B5CF6?style=for-the-badge&labelColor=gray" alt="purego"></a>
  <a href="https://github.com/younland/onnxruntime/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/younland/onnxruntime/ci.yml?style=for-the-badge&labelColor=gray&label=CI" alt="CI"></a>
  <a href="https://github.com/microsoft/onnxruntime"><img src="https://img.shields.io/badge/ORT-1.23~1.31-76B900?style=for-the-badge&labelColor=gray" alt="ORT"></a>
</p>

<p>
  <a href="#-核心特性">核心特性</a> ·
  <a href="#-应用截图">应用截图</a> ·
  <a href="#-安装">安装</a> ·
  <a href="#-快速开始">快速开始</a> ·
  <a href="#-配置说明">配置说明</a> ·
  <a href="#-开发指南">开发指南</a> ·
  <a href="#-功能细节">功能细节</a> ·
  <a href="#-性能基准">性能基准</a>
</p>

</div>

## ✨ 核心特性

### 🎯 运行时版本协商，一套代码跑 1.23 ~ 1.31

启动时读取 `OrtGetApiBase()->GetVersionString()` 解析真实版本，按 `GetAPI(min(minor, 31))` 获取与运行时匹配的 `OrtApi` 函数表，只调用该版本实际存在的函数。索引边界表 `apiLastIndex` 由 `scripts/gen_api_go.py` 从 v1.23.2 ~ v1.31.1 九个官方头文件逐一解析生成并校验"只追加"不变量，杜绝越界野指针。

| 运行时版本 | 行为 |
|---|---|
| < 1.23 | 拒绝启动 |
| 1.23 ~ 1.31 | 正常工作（CPU 推理全版本实测） |
| > 1.31（含 2.x） | 默认警告后放行；`WithStrictVersion(true)` 拒绝启动 |

### ⚡ CPU 全版本，CUDA / TensorRT 条件启用

CPU EP 内置恒可用；GPU EP 走双重门控——`IsProviderAvailable` 判断库构建是否包含该 EP，`EnableCUDA` / `EnableTensorRT` 把 ORT 的 status 错误（缺 CUDA 运行库、驱动不匹配等）原样透出。

### 🖥️ 跨平台 x64 + arm64

Windows x64/arm64、Linux x64/aarch64、macOS arm64/x86_64 统一一套源码；动态库加载 Windows 用 `syscall.LoadLibrary`、Unix 用 `purego.Dlopen`，CI 在 4 种 runner × 9 个 ORT 版本矩阵上全量验证。

### 📦 纯 Go 无 CGO

只依赖 `github.com/ebitengine/purego`，交叉编译无需 C 工具链；onnxruntime 动态库在运行时才加载，可按需替换 CPU/GPU 构建而不重新编译业务程序。

## 📷 应用截图

<div align="center">

<img src="docs/images/bus.jpg" width="45%" alt="输入图片 bus.jpg">　<img src="docs/images/yolov11_det.jpg" width="45%" alt="YOLO11 检测结果">

YOLO11 目标检测示例（[examples/yolo](examples/yolo)，ORT 1.31.1 CPU 推理）：1 辆 bus（class 5, 0.90）+ 4 个 person（class 0, 0.52~0.87）

</div>

## 📦 安装

```bash
go get github.com/younland/onnxruntime
```

把 onnxruntime 官方动态库放到 `lib/`（CPU 版从 [ORT Releases](https://github.com/microsoft/onnxruntime/releases) 下载；GPU 构建需自备 CUDA/TensorRT 依赖环境）：

| 平台 | 库文件 | 推荐路径 |
|---|---|---|
| Windows x64 / arm64 | `onnxruntime.dll` | `./lib/onnxruntime.dll` |
| Linux x64 / aarch64 | `libonnxruntime.so` | `./lib/onnxruntime_<arch>.so` |
| macOS arm64 / x86_64 | `libonnxruntime.dylib` | `./lib/onnxruntime_<arch>.dylib` |

`DefaultLibraryPath()` 按 `GOOS/GOARCH` 返回上述默认路径，也可用 `WithLibraryPath` 或 `ORT_LIB_PATH` 指定任意位置。

## 🚀 快速开始

```go
package main

import (
	"fmt"

	ort "github.com/younland/onnxruntime"
)

func main() {
	// 加载动态库：低于 1.23 拒绝启动；高于 1.31 默认警告放行，可 WithStrictVersion(true) 拒绝
	engine, err := ort.NewEngine(ort.WithLibraryPath(ort.DefaultLibraryPath()))
	if err != nil {
		panic(err)
	}
	defer engine.Destroy() // 最先 defer，最后执行；须晚于所有 Session/Value 的释放

	fmt.Println(engine.VersionString(), engine.APIVersion())

	opts, _ := engine.NewSessionOptions()
	defer opts.Destroy()
	opts.SetIntraOpNumThreads(4)

	// CUDA / TensorRT：按实际构建与依赖环境条件启用
	if engine.IsProviderAvailable(ort.ProviderCUDA) {
		if err := opts.EnableCUDA(nil); err != nil {
			fmt.Println("CUDA enable failed:", err) // 依赖缺失时错误原样透出
		}
	}

	sess, err := engine.NewSession("model.onnx", opts)
	if err != nil {
		panic(err)
	}
	defer sess.Destroy()

	input, _ := ort.NewTensor([]int64{1, 3, 224, 224}, make([]float32, 3*224*224))
	defer input.Destroy()

	outputs, err := sess.Run(map[string]*ort.Value{"input": input})
	if err != nil {
		panic(err)
	}
	for name, out := range outputs {
		defer out.Destroy()
		data, _ := ort.GetTensorData[float32](out)
		fmt.Println(name, len(data))
	}
}
```

资源释放顺序：`Value` / `Session` 由 `Env` 创建，必须先释放所有 `Session` 与 `Value`，再调用 `Engine.Destroy()`。与官方绑定一致，正常流程不卸载动态库本身（避免 ORT 静态析构崩溃）。

## ⚙️ 配置说明

| 环境变量 | 作用 | 生效位置 |
|---|---|---|
| `ORT_LIB_PATH` | 指定 onnxruntime 动态库完整路径；测试与示例优先读取，设置后跳过 `lib/` 版本矩阵 | 测试、示例程序 |

## 🛠️ 开发指南

```bash
# 单元测试（版本策略矩阵、索引边界，无需动态库）
go test ./...

# 拉取 1.23 ~ 1.31 全部 9 个版本实库 + yolo 模型/图片（约 300MB）
powershell -ExecutionPolicy Bypass -File scripts/fetch_deps.ps1   # Windows
./scripts/fetch_deps.sh                                           # Linux / macOS / Git Bash

# 端到端：yolo 测试对 lib/ 下每个版本各跑一个子测试；benchmark 同理
go test -v ./examples/yolo/
go test -bench . -benchmem ./examples/yolo/

# 重新生成 / 校验 api.go（依据官方头文件，含 --check 模式）
python3 scripts/gen_api_go.py [--check]
```

目录结构：

```text
api.go                  # gen_api_go.py 生成：OrtApi 结构体映射 + 版本索引边界表
engine.go               # 引擎加载、版本门禁、函数绑定
version.go              # 版本解析与最低/最高版本策略
funcs.go                # apiFuncs 函数表 + 索引可用性校验
session.go              # 会话与输入输出元信息
value.go                # Tensor 创建与读取
options.go              # 会话选项（线程、图优化、Arena、动态维度）
provider.go             # EP 探测与 CUDA/TensorRT 启用
internal/sys/           # windows: syscall.LoadLibrary / unix: purego.Dlopen
scripts/                # gen_api_go.py、fetch_deps.ps1|.sh
examples/yolo/          # YOLO11 检测示例（命令行 + 单元测试 + 基准测试）
docs/images/            # README 展示图片
.github/workflows/ci.yml
```

## 🧩 功能细节

### Execution Provider

| EP | 名称常量 | 支持情况 |
|---|---|---|
| CPU | `ProviderCPU` | 内置，始终可用，1.23 ~ 1.31 全版本实测 |
| CUDA | `ProviderCUDA` | 条件启用：库构建包含且 CUDA 依赖环境齐备 |
| TensorRT | `ProviderTensorRT` | 条件启用：同上，TensorRT 库需与 ORT 构建匹配 |

### 会话选项（SessionOptions）

| 方法 | 作用 |
|---|---|
| `SetIntraOpNumThreads` / `SetInterOpNumThreads` | 算子内 / 间并行线程数 |
| `SetGraphOptimizationLevel` | 图优化级别（None / Basic / Extended / All） |
| `SetExecutionMode` | 顺序 / 并行执行模式 |
| `SetCpuMemArena` | CPU 内存池开关（关：Destroy 即还内存；开：复用更快） |
| `AddFreeDimensionOverride[ByName]` | 按符号 / 维度名固化动态维度 |
| `EnableCUDA` / `EnableTensorRT` | 挂载 GPU EP，接受 `map[string]string` 键值配置或 nil 默认 |

### CI

GitHub Actions（[ci.yml](.github/workflows/ci.yml)，Go 1.27）：

| 任务 | 内容 |
|---|---|
| unit | `gen_api_go.py --check` + 单元测试（无需动态库） |
| e2e 矩阵 | 4 runner（windows x64 / linux x64 / linux aarch64 / macOS arm64）× 9 个 ORT 版本，每组合跑完整 `go test ./...`，Linux/macOS 加 `-race` |

## 📊 性能基准

环境：Windows 11 · Intel Core i5-14400（16 逻辑核）· Go 1.25 · yolo11n.onnx @ 640×640 · 输入 bus.jpg 810×1080 · ORT 官方 CPU 构建 · 每版本独立进程、预热后 10 次迭代均值。

**端到端（预处理 + 推理 + NMS）与推理**

| ORT 版本 | API | 推理 (ms) | 端到端 (ms) | 进程CPU/次 (ms) | CPU利用率 | 峰值RSS (MB) |
|---|---|---|---|---|---|---|
| 1.23.2 | 23 | 30.1 | 29.6 | 298 | ~1007% | 149 |
| 1.24.4 | 24 | 28.4 | 28.7 | 295 | ~1029% | 153 |
| 1.25.1 | 25 | 25.8 | 25.5 | 264 | ~1034% | 160 |
| 1.26.0 | 26 | 24.8 | 25.3 | 252 | ~993% | 159 |
| 1.27.1 | 27 | 24.2 | 24.0 | 250 | ~1040% | 159 |
| 1.28.3 | 28 | 23.6 | 24.1 | 250 | ~1037% | 159 |
| 1.29.1 | 29 | 27.3 | 27.9 | 277 | ~991% | 161 |
| 1.30.1 | 30 | 24.3 | 24.3 | 244 | ~1005% | 161 |
| 1.31.1 | 31 | 22.6 | 23.9 | 233 | ~974% | 161 |

**与 ORT 版本无关的纯 Go 阶段**（1.31.1 上实测，各版本一致）

| 阶段 | 耗时/次 | 分配 |
|---|---|---|
| 预处理（最近邻缩放 + CHW 归一化，零分配快路径） | ~2.6ms | ~4.9MB / 14 次 |
| 后处理（输出解析 + NMS） | ~5.3ms | ~9KB / 13 次 |

> [!NOTE]
> **1.25 起推理比 1.23/1.24 快约 15%**，与 ORT 官方 1.25 的 CPU 性能改进一致；1.26~1.31 基本持平（±5% 噪声内）。绝对值随硬件与负载波动，跨版本对比关注相对趋势。

> [!TIP]
> CPU 利用率 ~1000% 表示 ORT 线程池使用了约 10 个核；追求单请求低延迟可设 `opts.SetIntraOpNumThreads(1)`。单进程跑全版本矩阵时 RSS 为累计值，上表 RSS 为独立进程隔离测量。预处理已按解码类型（JPEG→YCbCr、PNG→RGBA/NRGBA）走零分配快路径，若输入是其它格式可先解码成这两类以获得同等性能。

## 🔗 参考

- [onnxruntime C API](https://onnxruntime.ai/docs/api/c/)
- [purego](https://github.com/ebitengine/purego)
- 设计参考 [onnxruntime_purego](https://github.com/getcharzp/onnxruntime_purego)
