//go:build linux || darwin

package main

import (
	"runtime"
	"syscall"
)

// processCPUTime 返回当前进程累计的用户态 + 内核态 CPU 时间（秒）。
// 包含进程内所有线程（含 ORT 线程池），因此多线程并行推理时可能超过墙钟时间。
func processCPUTime() float64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return timevalSeconds(ru.Utime) + timevalSeconds(ru.Stime)
}

func timevalSeconds(tv syscall.Timeval) float64 {
	return float64(tv.Sec) + float64(tv.Usec)*1e-6
}

// processRSS 返回当前 / 峰值驻留内存（字节）。
// linux 下 getrusage 的 Maxrss 单位为 KB，darwin 为字节。
func processRSS() (current, peak uintptr) {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0, 0
	}
	rss := uintptr(ru.Maxrss)
	if runtime.GOOS != "darwin" {
		rss *= 1024
	}
	return rss, rss
}
