package main

import (
	"fmt"
	"image"
	"os"
	"runtime"
	"testing"

	ort "github.com/younland/onnxruntime"
)

// setupBenchmark 用指定版本的库加载引擎 / 会话 / 测试图片；资源缺失时 b.Skip。
// 会话在多次迭代间复用（与生产用法一致），模型加载开销不计入基准。
func setupBenchmark(b *testing.B, lib testLibrary) (*ort.Engine, *ort.Session, image.Image) {
	b.Helper()

	model := "testdata/yolo11n.onnx"
	if _, err := os.Stat(model); err != nil {
		b.Skipf("model %s not found; run scripts/fetch_deps.ps1|.sh", model)
	}
	imagePath := findTestImage(b)

	engine, err := ort.NewEngine(ort.WithLibraryPath(lib.path))
	if err != nil {
		b.Fatalf("NewEngine(%s) failed: %v", lib.path, err)
	}
	opts, err := engine.NewSessionOptions()
	if err != nil {
		engine.Destroy()
		b.Fatalf("NewSessionOptions failed: %v", err)
	}
	sess, err := engine.NewSession(model, opts)
	opts.Destroy()
	if err != nil {
		engine.Destroy()
		b.Fatalf("NewSession failed: %v", err)
	}

	f, err := os.Open(imagePath)
	if err != nil {
		sess.Destroy()
		engine.Destroy()
		b.Fatalf("open image: %v", err)
	}
	img, err := decodeImage(f, imagePath)
	f.Close()
	if err != nil {
		sess.Destroy()
		engine.Destroy()
		b.Fatalf("decode image: %v", err)
	}

	b.Cleanup(func() {
		sess.Destroy()
		engine.Destroy()
	})

	b.Logf("library=%s version=%s api=%d GOMAXPROCS=%d NumCPU=%d",
		lib.path, engine.VersionString(), engine.APIVersion(), runtime.GOMAXPROCS(-1), runtime.NumCPU())
	return engine, sess, img
}

// runOnce 执行一次完整检测流水线（预处理 -> 推理 -> 后处理），并释放中间资源。
func runOnce(b *testing.B, engine *ort.Engine, sess *ort.Session, img image.Image) {
	b.Helper()

	input, params, err := preprocess(engine, img, 640)
	if err != nil {
		b.Fatalf("preprocess: %v", err)
	}
	outputs, err := sess.Run(map[string]*ort.Value{"images": input})
	input.Destroy()
	if err != nil {
		b.Fatalf("run: %v", err)
	}
	defer func() {
		for _, out := range outputs {
			out.Destroy()
		}
	}()

	output := outputs["output0"]
	if output == nil {
		b.Fatal("output0 missing")
	}
	data, err := ort.GetTensorData[float32](output)
	if err != nil {
		b.Fatalf("GetTensorData: %v", err)
	}
	shape, err := output.GetShape()
	if err != nil {
		b.Fatalf("GetShape: %v", err)
	}
	if _, err := postprocess(data, shape, params, 0.45, 0.5); err != nil {
		b.Fatalf("postprocess: %v", err)
	}
}

// forEachLibrary 对每个已下载的 onnxruntime 版本（1.23 ~ 1.31）跑一个子基准。
func forEachLibrary(b *testing.B, fn func(b *testing.B, lib testLibrary)) {
	for _, lib := range findTestLibraries(b) {
		b.Run(lib.version, func(b *testing.B) {
			fn(b, lib)
		})
	}
}

// BenchmarkYoloPreprocess 仅预处理：解码后的图片 -> 归一化 CHW Tensor。
func BenchmarkYoloPreprocess(b *testing.B) {
	forEachLibrary(b, func(b *testing.B, lib testLibrary) {
		engine, _, img := setupBenchmark(b, lib)
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			t, _, err := preprocess(engine, img, 640)
			if err != nil {
				b.Fatalf("preprocess: %v", err)
			}
			t.Destroy()
		}
	})
}

// BenchmarkYoloInference 仅模型推理（含输入张量创建与输出释放）。
// 预热一轮触发 ORT 线程池与内存池初始化。
func BenchmarkYoloInference(b *testing.B) {
	forEachLibrary(b, func(b *testing.B, lib testLibrary) {
		engine, sess, img := setupBenchmark(b, lib)

		input, _, err := preprocess(engine, img, 640)
		if err != nil {
			b.Fatalf("preprocess: %v", err)
		}
		warm, err := sess.Run(map[string]*ort.Value{"images": input})
		input.Destroy()
		if err != nil {
			b.Fatalf("warm-up run: %v", err)
		}
		for _, out := range warm {
			out.Destroy()
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			input, _, err := preprocess(engine, img, 640)
			if err != nil {
				b.Fatalf("preprocess: %v", err)
			}
			outputs, err := sess.Run(map[string]*ort.Value{"images": input})
			input.Destroy()
			if err != nil {
				b.Fatalf("run: %v", err)
			}
			for _, out := range outputs {
				out.Destroy()
			}
		}
	})
}

// BenchmarkYoloPostprocess 仅后处理：输出张量解析 + NMS。
func BenchmarkYoloPostprocess(b *testing.B) {
	forEachLibrary(b, func(b *testing.B, lib testLibrary) {
		engine, sess, img := setupBenchmark(b, lib)

		input, params, err := preprocess(engine, img, 640)
		if err != nil {
			b.Fatalf("preprocess: %v", err)
		}
		outputs, err := sess.Run(map[string]*ort.Value{"images": input})
		input.Destroy()
		if err != nil {
			b.Fatalf("run: %v", err)
		}
		output := outputs["output0"]
		data, err := ort.GetTensorData[float32](output)
		if err != nil {
			b.Fatalf("GetTensorData: %v", err)
		}
		shape, err := output.GetShape()
		if err != nil {
			b.Fatalf("GetShape: %v", err)
		}
		for _, out := range outputs {
			out.Destroy()
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if _, err := postprocess(data, shape, params, 0.45, 0.5); err != nil {
				b.Fatalf("postprocess: %v", err)
			}
		}
	})
}

// BenchmarkYoloEndToEnd 端到端基准：在 ns/op 之外报告
// cpuns/op（进程 CPU 时间）、heapMB/op（Go 堆增量）、peakRSSMB（进程峰值工作集）。
//
// 注意：ORT 线程池的 CPU 消耗计入 cpuns/op，因此其值可能大于墙钟 ns/op
// （比例即并行度）；RSS 为进程级，含 ORT 原生内存。
func BenchmarkYoloEndToEnd(b *testing.B) {
	forEachLibrary(b, func(b *testing.B, lib testLibrary) {
		engine, sess, img := setupBenchmark(b, lib)

		// 预热一轮，排除线程池 / 内存池初始化造成的尖峰
		runOnce(b, engine, sess, img)

		var mStart, mEnd runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&mStart)
		cpuStart := processCPUTime()

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			runOnce(b, engine, sess, img)
		}
		b.StopTimer()

		cpuEnd := processCPUTime()
		_, rssPeak := processRSS()
		runtime.ReadMemStats(&mEnd)

		n := float64(b.N)
		b.ReportMetric((cpuEnd-cpuStart)*1e9/n, "cpuns/op")
		b.ReportMetric(float64(mEnd.HeapInuse-mStart.HeapInuse)/1e6/n, "heapMB/op")
		b.ReportMetric(float64(rssPeak)/1e6, "peakRSSMB")

		fmt.Printf("  total: wall=%.2fs cpu=%.2fs cpuUtilization=%.0f%% peakRSS=%.0fMB iterations=%d\n",
			b.Elapsed().Seconds(), cpuEnd-cpuStart,
			(cpuEnd-cpuStart)/b.Elapsed().Seconds()*100,
			float64(rssPeak)/1e6, b.N)
	})
}
