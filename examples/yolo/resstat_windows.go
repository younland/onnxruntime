//go:build windows

package main

import (
	"syscall"
	"unsafe"
)

var (
	kernel32 = syscall.NewLazyDLL("kernel32.dll")
	psapi    = syscall.NewLazyDLL("psapi.dll")

	getProcessTimes      = kernel32.NewProc("GetProcessTimes")
	getProcessMemoryInfo = psapi.NewProc("GetProcessMemoryInfo")
)

// currentProcess 是 GetCurrentProcess() 返回的伪句柄，可被 GetProcessTimes / GetProcessMemoryInfo 直接使用。
const currentProcess = uintptr(0xFFFFFFFFFFFFFFFF)

type filetime struct {
	low  uint32
	high uint32
}

// seconds 把 FILETIME（100 纳秒间隔）转换为秒。
func (f filetime) seconds() float64 {
	v := uint64(f.high)<<32 | uint64(f.low)
	return float64(v) * 100e-9
}

// processCPUTime 返回当前进程累计的用户态 + 内核态 CPU 时间（秒）。
// 包含进程内所有线程（含 ORT 线程池），因此多线程并行推理时可能超过墙钟时间。
func processCPUTime() float64 {
	var creation, exit, kernel, user filetime
	r1, _, _ := getProcessTimes.Call(currentProcess,
		uintptr(unsafe.Pointer(&creation)),
		uintptr(unsafe.Pointer(&exit)),
		uintptr(unsafe.Pointer(&kernel)),
		uintptr(unsafe.Pointer(&user)),
	)
	if r1 == 0 {
		return 0
	}
	return kernel.seconds() + user.seconds()
}

// processMemoryCounters 对应 C 侧 PROCESS_MEMORY_COUNTERS（64 位布局）。
type processMemoryCounters struct {
	cb                         uint32
	pageFaultCount             uint32
	peakWorkingSetSize         uintptr
	workingSetSize             uintptr
	quotaPeakPagedPoolUsage    uintptr
	quotaPagedPoolUsage        uintptr
	quotaPeakNonPagedPoolUsage uintptr
	quotaNonPagedPoolUsage     uintptr
	pagefileUsage              uintptr
	peakPagefileUsage          uintptr
}

// processRSS 返回当前工作集与峰值工作集（字节）。
func processRSS() (current, peak uintptr) {
	var mc processMemoryCounters
	mc.cb = uint32(unsafe.Sizeof(mc))
	r1, _, _ := getProcessMemoryInfo.Call(currentProcess,
		uintptr(unsafe.Pointer(&mc)), uintptr(mc.cb))
	if r1 == 0 {
		return 0, 0
	}
	return mc.workingSetSize, mc.peakWorkingSetSize
}
