package ort

import (
	"fmt"
	"unsafe"
)

// 常用 Execution Provider 名称常量。
const (
	ProviderCPU      = "CPUExecutionProvider"
	ProviderCUDA     = "CUDAExecutionProvider"
	ProviderTensorRT = "TensorrtExecutionProvider"
)

// providerOptions 描述一个 V2 形式的 Execution Provider：创建配置 -> 写入键值 -> 挂到 SessionOptions。
type providerOptions[H any] struct {
	create  func(*H) StatusHandle
	update  func(H, **byte, **byte, uintptr) StatusHandle
	release func(H)
	append  func(SessionOptionsHandle, H) StatusHandle
}

func (f *apiFuncs) cudaOptions() providerOptions[CUDAProviderOptionsV2Handle] {
	return providerOptions[CUDAProviderOptionsV2Handle]{
		create:  f.createCUDAProviderOptions,
		update:  f.updateCUDAProviderOptions,
		release: f.releaseCUDAProviderOptions,
		append:  f.appendExecutionProvider_CUDA_V2,
	}
}

func (f *apiFuncs) tensorRTOptions() providerOptions[TensorRTProviderOptionsV2Handle] {
	return providerOptions[TensorRTProviderOptionsV2Handle]{
		create:  f.createTensorRTProviderOptions,
		update:  f.updateTensorRTProviderOptions,
		release: f.releaseTensorRTProviderOptions,
		append:  f.appendExecutionProvider_TensorRT_V2,
	}
}

// EnableCUDA 启用 CUDA，options 为 CUDA Provider 的键值配置，可为 nil。
//
// 启用前可用 IsProviderAvailable(ProviderCUDA) 判断当前库是否包含 CUDA EP；
// 依赖环境（CUDA 运行库、驱动）不满足时，错误信息原样透出。
//
// 常用键：
//
//	device_id                    = "0"                指定显卡，默认 0
//	cudnn_conv_algo_search       = "HEURISTIC"        卷积算法搜索，默认 EXHAUSTIVE
//	cudnn_conv_use_max_workspace = "1"                卷积使用最大 workspace，默认 1
//	enable_cuda_graph            = "1"                输入 shape 固定时启用 CUDA Graph，默认 0
//	gpu_mem_limit                = "4294967296"       显存上限（字节），默认不限制
//	arena_extend_strategy        = "kSameAsRequested" 显存池扩展策略，默认 kNextPowerOfTwo
func (o *SessionOptions) EnableCUDA(options map[string]string) error {
	p := o.engine.funcs.cudaOptions()
	if p.create == nil || p.update == nil || p.release == nil || p.append == nil {
		return fmt.Errorf("CUDA Execution Provider is not supported by onnxruntime %s", o.engine.version)
	}
	return enableProvider(o, p, options)
}

// EnableTensorRT 启用 TensorRT，options 为 TensorRT Provider 的键值配置，可为 nil。
//
// 与 CUDA 一样用 device_id 指定显卡，从 0 开始计数，默认 0：
//
//	opts.EnableTensorRT(map[string]string{"device_id": "1"})
//
// 常用键：
//
//	trt_fp16_enable         = "1"     启用 FP16，CV 推理加速明显
//	trt_int8_enable         = "1"     启用 INT8
//	trt_engine_cache_enable = "1"     缓存引擎，避免每次构建耗时
//	trt_engine_cache_path   = "./trt" 引擎缓存目录
//
// Execution Provider 按挂载顺序优先生效，需同时使用 CUDA 兜底时先调用本方法，
// 且两者的 device_id 应指向同一张卡：
//
//	opts.EnableTensorRT(map[string]string{"device_id": "1", "trt_fp16_enable": "1"})
//	opts.EnableCUDA(map[string]string{"device_id": "1"})
func (o *SessionOptions) EnableTensorRT(options map[string]string) error {
	p := o.engine.funcs.tensorRTOptions()
	if p.create == nil || p.update == nil || p.release == nil || p.append == nil {
		return fmt.Errorf("TensorRT Execution Provider is not supported by onnxruntime %s", o.engine.version)
	}
	return enableProvider(o, p, options)
}

// enableProvider 向 SessionOptions 追加一个 Execution Provider 并写入其键值配置。
func enableProvider[H any](o *SessionOptions, p providerOptions[H], options map[string]string) error {
	var cfg H
	if err := o.engine.checkStatus(p.create(&cfg)); err != nil {
		return fmt.Errorf("failed to create provider options: %w", err)
	}
	defer p.release(cfg)

	if len(options) > 0 {
		keys := make([]*byte, 0, len(options))
		values := make([]*byte, 0, len(options))
		for k, v := range options {
			key, _ := stringToCString(k)
			value, _ := stringToCString(v)
			keys, values = append(keys, key), append(values, value)
		}
		if err := o.engine.checkStatus(p.update(cfg, slicePtr(keys), slicePtr(values), uintptr(len(keys)))); err != nil {
			return fmt.Errorf("failed to update provider options: %w", err)
		}
	}

	return o.engine.checkStatus(p.append(o.handle, cfg))
}

// GetAvailableProviders 返回当前 onnxruntime 构建支持的 Execution Provider 名称列表。
//
// 具体返回范围取决于编译时启用的 EP：CPU 始终存在；CUDA / TensorRT 仅存在于
// GPU 构建中。查询失败时返回 nil。
func (e *Engine) GetAvailableProviders() []string {
	if e.funcs.getAvailableProviders == nil || e.funcs.releaseAvailableProviders == nil {
		return nil
	}
	var providers **byte
	var count int32
	if err := e.checkStatus(e.funcs.getAvailableProviders(&providers, &count)); err != nil {
		return nil
	}
	defer e.funcs.releaseAvailableProviders(providers, count)

	names := make([]string, 0, count)
	if providers == nil {
		return names
	}
	for _, p := range unsafe.Slice(providers, count) {
		names = append(names, cStringToString(p))
	}
	return names
}

// IsProviderAvailable 判断当前 onnxruntime 构建是否包含指定 Execution Provider。
//
//	if engine.IsProviderAvailable(ort.ProviderCUDA) {
//	    opts.EnableCUDA(nil)
//	}
func (e *Engine) IsProviderAvailable(name string) bool {
	for _, p := range e.GetAvailableProviders() {
		if p == name {
			return true
		}
	}
	return false
}
