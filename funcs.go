package ort

import (
	"unsafe"

	"github.com/ebitengine/purego"
)

// apiFuncs 保存 purego 绑定后的 Go 可调用函数。
// 任何字段在使用前都必须通过 engine.available(idx) 校验：低版本运行时
// 返回的 OrtApi 表更短，越界字段是无效的野指针。
type apiFuncs struct {
	getVersionString func() string

	// status, error
	createStatus    func(ErrorCode, *byte) StatusHandle
	getErrorCode    func(StatusHandle) ErrorCode
	getErrorMessage func(StatusHandle) unsafe.Pointer
	releaseStatus   func(StatusHandle)

	// env
	createEnv  func(LoggingLevel, *byte, *EnvHandle) StatusHandle
	releaseEnv func(EnvHandle)

	// allocator
	getAllocatorWithDefaultOptions func(*AllocatorHandle) StatusHandle
	allocatorFree                  func(AllocatorHandle, unsafe.Pointer)

	// memory info
	createCpuMemoryInfo func(AllocatorType, MemType, *MemoryInfoHandle) StatusHandle
	releaseMemoryInfo   func(MemoryInfoHandle)

	// CUDA
	createCUDAProviderOptions       func(*CUDAProviderOptionsV2Handle) StatusHandle
	updateCUDAProviderOptions       func(CUDAProviderOptionsV2Handle, **byte, **byte, uintptr) StatusHandle
	releaseCUDAProviderOptions      func(CUDAProviderOptionsV2Handle)
	appendExecutionProvider_CUDA_V2 func(SessionOptionsHandle, CUDAProviderOptionsV2Handle) StatusHandle

	// TensorRT
	createTensorRTProviderOptions       func(*TensorRTProviderOptionsV2Handle) StatusHandle
	updateTensorRTProviderOptions       func(TensorRTProviderOptionsV2Handle, **byte, **byte, uintptr) StatusHandle
	releaseTensorRTProviderOptions      func(TensorRTProviderOptionsV2Handle)
	appendExecutionProvider_TensorRT_V2 func(SessionOptionsHandle, TensorRTProviderOptionsV2Handle) StatusHandle

	// session options
	createSessionOptions             func(*SessionOptionsHandle) StatusHandle
	setIntraOpNumThreads             func(SessionOptionsHandle, int32) StatusHandle
	setInterOpNumThreads             func(SessionOptionsHandle, int32) StatusHandle
	setSessionExecutionMode          func(SessionOptionsHandle, int32) StatusHandle
	setSessionGraphOptimizationLevel func(SessionOptionsHandle, int32) StatusHandle
	setSessionLogSeverityLevel       func(SessionOptionsHandle, int32) StatusHandle
	releaseSessionOptions            func(SessionOptionsHandle)
	enableCpuMemArena                func(SessionOptionsHandle) StatusHandle
	disableCpuMemArena               func(SessionOptionsHandle) StatusHandle
	addFreeDimensionOverride         func(SessionOptionsHandle, *byte, int64) StatusHandle
	addFreeDimensionOverrideByName   func(SessionOptionsHandle, *byte, int64) StatusHandle

	// session
	createSession            func(EnvHandle, unsafe.Pointer, SessionOptionsHandle, *SessionHandle) StatusHandle
	createSessionFromArray   func(EnvHandle, unsafe.Pointer, uintptr, SessionOptionsHandle, *SessionHandle) StatusHandle
	sessionGetInputCount     func(SessionHandle, *uintptr) StatusHandle
	sessionGetOutputCount    func(SessionHandle, *uintptr) StatusHandle
	sessionGetInputName      func(SessionHandle, uintptr, AllocatorHandle, **byte) StatusHandle
	sessionGetOutputName     func(SessionHandle, uintptr, AllocatorHandle, **byte) StatusHandle
	sessionGetInputTypeInfo  func(SessionHandle, uintptr, *TypeInfoHandle) StatusHandle
	sessionGetOutputTypeInfo func(SessionHandle, uintptr, *TypeInfoHandle) StatusHandle
	castTypeInfoToTensorInfo func(TypeInfoHandle, *TensorTypeAndShapeInfoHandle) StatusHandle
	releaseTypeInfo          func(TypeInfoHandle)
	run                      func(SessionHandle, uintptr, *unsafe.Pointer, *ValueHandle, uintptr, *unsafe.Pointer, uintptr, *ValueHandle) StatusHandle
	releaseSession           func(SessionHandle)

	// tensor, value
	createTensorWithDataAsOrtValue func(MemoryInfoHandle, unsafe.Pointer, uintptr, *int64, uintptr, TensorElementDataType, *ValueHandle) StatusHandle
	getValueType                   func(ValueHandle, *OnnxType) StatusHandle
	getTensorMutableData           func(ValueHandle, *unsafe.Pointer) StatusHandle
	getTensorTypeAndShape          func(ValueHandle, *TensorTypeAndShapeInfoHandle) StatusHandle
	getTensorElementType           func(TensorTypeAndShapeInfoHandle, *TensorElementDataType) StatusHandle
	getDimensionsCount             func(TensorTypeAndShapeInfoHandle, *uintptr) StatusHandle
	getDimensions                  func(TensorTypeAndShapeInfoHandle, *int64, uintptr) StatusHandle
	getTensorShapeElementCount     func(TensorTypeAndShapeInfoHandle, *uintptr) StatusHandle
	releaseValue                   func(ValueHandle)
	releaseTensorTypeAndShapeInfo  func(TensorTypeAndShapeInfoHandle)

	// provider
	getAvailableProviders     func(***byte, *int32) StatusHandle
	releaseAvailableProviders func(**byte, int32) StatusHandle
}

// OrtApi 表中各函数的索引，用于运行时可用性校验（见 api.go 中的注释序号）。
const (
	idxCreateStatus                     = 0
	idxGetErrorCode                     = 1
	idxGetErrorMessage                  = 2
	idxCreateEnv                        = 3
	idxCreateSession                    = 7
	idxCreateSessionFromArray           = 8
	idxRun                              = 9
	idxCreateSessionOptions             = 10
	idxSetSessionExecutionMode          = 13
	idxEnableCpuMemArena                = 18
	idxDisableCpuMemArena               = 19
	idxSetSessionLogSeverityLevel       = 22
	idxSetSessionGraphOptimizationLevel = 23
	idxSetIntraOpNumThreads             = 24
	idxSetInterOpNumThreads             = 25
	idxSessionGetInputCount             = 30
	idxSessionGetOutputCount            = 31
	idxSessionGetInputTypeInfo          = 33
	idxSessionGetOutputTypeInfo         = 34
	idxSessionGetInputName              = 36
	idxSessionGetOutputName             = 37
	idxCreateTensorWithDataAsOrtValue   = 49
	idxGetTensorMutableData             = 51
	idxCastTypeInfoToTensorInfo         = 55
	idxGetTensorElementType             = 60
	idxGetDimensionsCount               = 61
	idxGetDimensions                    = 62
	idxGetTensorShapeElementCount       = 64
	idxGetTensorTypeAndShape            = 65
	idxGetValueType                     = 67
	idxCreateCpuMemoryInfo              = 69
	idxAllocatorFree                    = 76
	idxGetAllocatorWithDefaultOptions   = 78
	idxAddFreeDimensionOverride         = 79
	idxAddFreeDimensionOverrideByName   = 80
	idxReleaseEnv                       = 92
	idxReleaseStatus                    = 93
	idxReleaseMemoryInfo                = 94
	idxReleaseSession                   = 95
	idxReleaseValue                     = 96
	idxReleaseTypeInfo                  = 98
	idxReleaseTensorTypeAndShapeInfo    = 99
	idxReleaseSessionOptions            = 100
	idxGetAvailableProviders            = 125
	idxReleaseAvailableProviders        = 126
	idxAppendExecutionProviderTRTV2     = 170
	idxCreateTensorRTProviderOptions    = 171
	idxUpdateTensorRTProviderOptions    = 172
	idxReleaseTensorRTProviderOptions   = 174
	idxAppendExecutionProviderCUDAV2    = 204
	idxCreateCUDAProviderOptions        = 205
	idxUpdateCUDAProviderOptions        = 206
	idxReleaseCUDAProviderOptions       = 208
)

// bindAPIFuncs 按运行时实际的 OrtApi 表版本绑定函数：索引超过 last 的一律跳过，
// 对应字段保持 nil，由调用方通过 available() 判断后给出友好错误。
func (e *Engine) bindAPIFuncs() {
	f := e.funcs
	a := e.api
	last := e.lastIndex

	bind := func(idx uint32, dst any, src uintptr) {
		if idx <= last && src != 0 {
			purego.RegisterFunc(dst, src)
		}
	}

	// 版本字符串走 OrtApiBase，不受表长度限制
	purego.RegisterFunc(&f.getVersionString, e.apiBase.GetVersionString)

	bind(idxCreateStatus, &f.createStatus, a.CreateStatus)
	bind(idxGetErrorCode, &f.getErrorCode, a.GetErrorCode)
	bind(idxGetErrorMessage, &f.getErrorMessage, a.GetErrorMessage)
	bind(idxReleaseStatus, &f.releaseStatus, a.ReleaseStatus)

	bind(idxCreateEnv, &f.createEnv, a.CreateEnv)
	bind(idxReleaseEnv, &f.releaseEnv, a.ReleaseEnv)

	bind(idxGetAllocatorWithDefaultOptions, &f.getAllocatorWithDefaultOptions, a.GetAllocatorWithDefaultOptions)
	bind(idxAllocatorFree, &f.allocatorFree, a.AllocatorFree)

	bind(idxCreateCpuMemoryInfo, &f.createCpuMemoryInfo, a.CreateCpuMemoryInfo)
	bind(idxReleaseMemoryInfo, &f.releaseMemoryInfo, a.ReleaseMemoryInfo)

	// CUDA（1.23 ~ 1.31 均包含以下全部索引）
	bind(idxCreateCUDAProviderOptions, &f.createCUDAProviderOptions, a.CreateCUDAProviderOptions)
	bind(idxUpdateCUDAProviderOptions, &f.updateCUDAProviderOptions, a.UpdateCUDAProviderOptions)
	bind(idxReleaseCUDAProviderOptions, &f.releaseCUDAProviderOptions, a.ReleaseCUDAProviderOptions)
	bind(idxAppendExecutionProviderCUDAV2, &f.appendExecutionProvider_CUDA_V2, a.SessionOptionsAppendExecutionProvider_CUDA_V2)

	// TensorRT
	bind(idxCreateTensorRTProviderOptions, &f.createTensorRTProviderOptions, a.CreateTensorRTProviderOptions)
	bind(idxUpdateTensorRTProviderOptions, &f.updateTensorRTProviderOptions, a.UpdateTensorRTProviderOptions)
	bind(idxReleaseTensorRTProviderOptions, &f.releaseTensorRTProviderOptions, a.ReleaseTensorRTProviderOptions)
	bind(idxAppendExecutionProviderTRTV2, &f.appendExecutionProvider_TensorRT_V2, a.SessionOptionsAppendExecutionProvider_TensorRT_V2)

	// session options
	bind(idxCreateSessionOptions, &f.createSessionOptions, a.CreateSessionOptions)
	bind(idxSetIntraOpNumThreads, &f.setIntraOpNumThreads, a.SetIntraOpNumThreads)
	bind(idxSetInterOpNumThreads, &f.setInterOpNumThreads, a.SetInterOpNumThreads)
	bind(idxSetSessionExecutionMode, &f.setSessionExecutionMode, a.SetSessionExecutionMode)
	bind(idxSetSessionGraphOptimizationLevel, &f.setSessionGraphOptimizationLevel, a.SetSessionGraphOptimizationLevel)
	bind(idxSetSessionLogSeverityLevel, &f.setSessionLogSeverityLevel, a.SetSessionLogSeverityLevel)
	bind(idxReleaseSessionOptions, &f.releaseSessionOptions, a.ReleaseSessionOptions)
	bind(idxEnableCpuMemArena, &f.enableCpuMemArena, a.EnableCpuMemArena)
	bind(idxDisableCpuMemArena, &f.disableCpuMemArena, a.DisableCpuMemArena)
	bind(idxAddFreeDimensionOverride, &f.addFreeDimensionOverride, a.AddFreeDimensionOverride)
	bind(idxAddFreeDimensionOverrideByName, &f.addFreeDimensionOverrideByName, a.AddFreeDimensionOverrideByName)

	// session
	bind(idxCreateSession, &f.createSession, a.CreateSession)
	bind(idxCreateSessionFromArray, &f.createSessionFromArray, a.CreateSessionFromArray)
	bind(idxSessionGetInputCount, &f.sessionGetInputCount, a.SessionGetInputCount)
	bind(idxSessionGetOutputCount, &f.sessionGetOutputCount, a.SessionGetOutputCount)
	bind(idxSessionGetInputName, &f.sessionGetInputName, a.SessionGetInputName)
	bind(idxSessionGetOutputName, &f.sessionGetOutputName, a.SessionGetOutputName)
	bind(idxSessionGetInputTypeInfo, &f.sessionGetInputTypeInfo, a.SessionGetInputTypeInfo)
	bind(idxSessionGetOutputTypeInfo, &f.sessionGetOutputTypeInfo, a.SessionGetOutputTypeInfo)
	bind(idxCastTypeInfoToTensorInfo, &f.castTypeInfoToTensorInfo, a.CastTypeInfoToTensorInfo)
	bind(idxReleaseTypeInfo, &f.releaseTypeInfo, a.ReleaseTypeInfo)
	bind(idxRun, &f.run, a.Run)
	bind(idxReleaseSession, &f.releaseSession, a.ReleaseSession)

	// tensor, value
	bind(idxCreateTensorWithDataAsOrtValue, &f.createTensorWithDataAsOrtValue, a.CreateTensorWithDataAsOrtValue)
	bind(idxGetValueType, &f.getValueType, a.GetValueType)
	bind(idxGetTensorMutableData, &f.getTensorMutableData, a.GetTensorMutableData)
	bind(idxGetTensorTypeAndShape, &f.getTensorTypeAndShape, a.GetTensorTypeAndShape)
	bind(idxGetTensorElementType, &f.getTensorElementType, a.GetTensorElementType)
	bind(idxGetDimensionsCount, &f.getDimensionsCount, a.GetDimensionsCount)
	bind(idxGetDimensions, &f.getDimensions, a.GetDimensions)
	bind(idxGetTensorShapeElementCount, &f.getTensorShapeElementCount, a.GetTensorShapeElementCount)
	bind(idxReleaseValue, &f.releaseValue, a.ReleaseValue)
	bind(idxReleaseTensorTypeAndShapeInfo, &f.releaseTensorTypeAndShapeInfo, a.ReleaseTensorTypeAndShapeInfo)

	// provider
	bind(idxGetAvailableProviders, &f.getAvailableProviders, a.GetAvailableProviders)
	bind(idxReleaseAvailableProviders, &f.releaseAvailableProviders, a.ReleaseAvailableProviders)
}

// available 判断索引 idx 的函数在当前运行时的 OrtApi 表中是否存在。
func (e *Engine) available(idx uint32) bool {
	return idx <= e.lastIndex
}

// require 校验一组核心函数已绑定（用于启动自检与功能入口）。
func (e *Engine) require(indices ...uint32) error {
	for _, idx := range indices {
		if !e.available(idx) {
			return errAPIUnavailable(idx)
		}
	}
	return nil
}
