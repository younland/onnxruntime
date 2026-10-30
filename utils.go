package ort

import (
	"fmt"
	"runtime"
	"unsafe"
)

// DefaultLibraryPath 默认动态库路径，例如：
//
//	Windows: ./lib/onnxruntime.dll
//	Linux amd64: ./lib/onnxruntime_amd64.so
//	Linux arm64: ./lib/onnxruntime_arm64.so
//	Mac amd64: ./lib/onnxruntime_amd64.dylib
//	Mac arm64: ./lib/onnxruntime_arm64.dylib
func DefaultLibraryPath() string {
	baseDir := "./lib/"
	libName := "onnxruntime"

	if runtime.GOOS == "windows" {
		return baseDir + libName + ".dll"
	}

	var ext string
	switch runtime.GOOS {
	case "darwin":
		ext = "dylib"
	case "linux":
		ext = "so"
	default:
		return baseDir + libName + "_amd64.so"
	}
	return fmt.Sprintf("%s%s_%s.%s", baseDir, libName, runtime.GOARCH, ext)
}

// slicePtr 返回切片首元素的地址。空切片返回 nil（C API 允许 nil 配长度 0）。
func slicePtr[T any](s []T) *T {
	if len(s) == 0 {
		return nil
	}
	return &s[0]
}

// stringToCString 将字符串转换为 C 字符串（带 NUL 结尾），返回指针。
//
// 调用方需保证指针在 C 调用完成前不被 GC：通常做法是让返回值在调用期间保持存活。
func stringToCString(s string) (*byte, error) {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return &b[0], nil
}

// cStringToString 将 C 字符串转换为 Go 字符串。
func cStringToString(ptr *byte) string {
	if ptr == nil {
		return ""
	}
	var length int
	for {
		if *(*byte)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(length))) == 0 {
			break
		}
		length++
	}
	return string(unsafe.Slice(ptr, length))
}
