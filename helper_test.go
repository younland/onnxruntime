package ort

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"testing"
)

// testLibName 当前平台下 onnxruntime 动态库的文件名。
func testLibName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
}

// findTestLibrary 定位用于端到端测试的动态库：
// 优先 ORT_LIB_PATH 环境变量，其次枚举 lib/<version>/ 下所有已下载版本（按版本排序取第一个）。
// 找不到时跳过测试（先运行 scripts/fetch_deps.ps1|.sh 或设置 ORT_LIB_PATH）。
func findTestLibrary(t testing.TB) string {
	t.Helper()

	if p := os.Getenv("ORT_LIB_PATH"); p != "" {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	name := testLibName()
	entries, err := os.ReadDir("lib")
	if err == nil {
		var versions []string
		for _, e := range entries {
			if e.IsDir() {
				versions = append(versions, e.Name())
			}
		}
		sort.Strings(versions)
		for _, v := range versions {
			p := filepath.Join("lib", v, name)
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}

	t.Skip("onnxruntime dynamic library not found; run scripts/fetch_deps.ps1|.sh or set ORT_LIB_PATH")
	return ""
}
