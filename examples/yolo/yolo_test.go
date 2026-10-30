package main

import (
	"image"
	"image/png"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"

	ort "github.com/younland/onnxruntime"
)

// testLibrary 一个可用于测试的 onnxruntime 动态库及其版本。
type testLibrary struct {
	version string // lib/<version> 目录名，如 "1.23.2"；来自 ORT_LIB_PATH 时为 "env"
	path    string
}

// libFileName 返回当前平台下 onnxruntime 动态库的文件名。
func libFileName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
}

// findTestLibraries 枚举测试库，覆盖 1.23 ~ 1.31 版本矩阵：
// 优先 ORT_LIB_PATH（单库），其次 lib/<version>/ 下所有已下载版本（按版本排序）。
// 一个都找不到时跳过测试（先运行 scripts/fetch_deps.ps1|.sh 或设置 ORT_LIB_PATH）。
func findTestLibraries(t testing.TB) []testLibrary {
	t.Helper()

	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return []testLibrary{{version: "env", path: p}}
		}
	}

	name := libFileName()
	var libs []testLibrary
	// 测试工作目录通常是 examples/yolo，库在仓库根的 lib/ 下
	for _, libRoot := range []string{"lib", filepath.Join("..", "..", "lib")} {
		entries, err := os.ReadDir(libRoot)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			p := filepath.Join(libRoot, e.Name(), name)
			if _, err := os.Stat(p); err == nil {
				libs = append(libs, testLibrary{version: e.Name(), path: p})
			}
		}
	}
	sort.Slice(libs, func(i, j int) bool { return libs[i].version < libs[j].version })

	if len(libs) == 0 {
		t.Skip("onnxruntime dynamic library not found; run scripts/fetch_deps.ps1|.sh or set ORT_LIB_PATH")
	}
	return libs
}

// findLibrary 返回第一个可用测试库（基准测试用单库，版本矩阵见 findTestLibraries）。
func findLibrary(t testing.TB) string {
	t.Helper()
	return findTestLibraries(t)[0].path
}

// findTestImage 定位测试图片。
func findTestImage(t testing.TB) string {
	t.Helper()
	for _, p := range []string{"testdata/bus.jpg", "testdata/yolo_demo.png", "testdata/yolo_demo.jpg"} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skip("test image not found in testdata/")
	return ""
}

// TestYolo11Detect 端到端 YOLO11 检测：预处理 -> CPU 推理 -> NMS 后处理。
// 对 lib/ 下每个已下载的 onnxruntime 版本（1.23 ~ 1.31）各跑一个子测试。
func TestYolo11Detect(t *testing.T) {
	model := "testdata/yolo11n.onnx"
	if _, err := os.Stat(model); err != nil {
		t.Skipf("model %s not found; run scripts/fetch_deps.ps1|.sh", model)
	}
	imagePath := findTestImage(t)

	f, err := os.Open(imagePath)
	if err != nil {
		t.Fatalf("open image: %v", err)
	}
	img, err := decodeImage(f, imagePath)
	f.Close()
	if err != nil {
		t.Fatalf("decode image: %v", err)
	}

	for _, lib := range findTestLibraries(t) {
		t.Run(lib.version, func(t *testing.T) {
			detectWithLibrary(t, lib, model, img)
		})
	}
}

// detectWithLibrary 用指定版本的动态库执行一次完整检测并校验结果。
func detectWithLibrary(t *testing.T, lib testLibrary, model string, img image.Image) {
	engine, err := ort.NewEngine(ort.WithLibraryPath(lib.path))
	if err != nil {
		t.Fatalf("NewEngine(%s) failed: %v", lib.path, err)
	}
	defer engine.Destroy()
	t.Logf("library=%s version=%s api=%d providers=%v",
		lib.path, engine.VersionString(), engine.APIVersion(), engine.GetAvailableProviders())

	dets, err := Detect(engine, model, img, 640, 0.45, 0.5)
	if err != nil {
		t.Fatalf("Detect failed: %v", err)
	}

	if len(dets) == 0 {
		t.Fatal("no detections; expected at least one object in the demo image")
	}
	for _, det := range dets {
		t.Logf("Class: %d, Score: %.2f, Box: %v", det.ClassID, det.Score, det.Box)
		if det.Score < 0.45 || det.Score > 1.0 {
			t.Errorf("score %f out of range", det.Score)
		}
		if det.Box.Empty() {
			t.Errorf("empty box for class %d", det.ClassID)
		}
	}

	// 保存标注结果，便于人工核对
	outPath := filepath.Join(t.TempDir(), "yolov11_det.png")
	out, err := os.Create(outPath)
	if err != nil {
		t.Fatalf("create output: %v", err)
	}
	if err := png.Encode(out, drawDetections(img, dets)); err != nil {
		t.Fatalf("encode output: %v", err)
	}
	out.Close()
	t.Logf("annotated image saved: %s", outPath)
}
