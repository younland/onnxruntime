package ort

// SessionOptions 会话配置。
type SessionOptions struct {
	handle SessionOptionsHandle
	engine *Engine
}

// NewSessionOptions 创建会话配置。
func (e *Engine) NewSessionOptions() (*SessionOptions, error) {
	if !e.available(idxCreateSessionOptions) {
		return nil, errAPIUnavailable(idxCreateSessionOptions)
	}
	var h SessionOptionsHandle
	if err := e.checkStatus(e.funcs.createSessionOptions(&h)); err != nil {
		return nil, err
	}
	return &SessionOptions{handle: h, engine: e}, nil
}

// SetIntraOpNumThreads 设置算子内并行线程数。
func (o *SessionOptions) SetIntraOpNumThreads(num int32) error {
	if o.engine.funcs.setIntraOpNumThreads == nil {
		return errAPIUnavailable(idxSetIntraOpNumThreads)
	}
	return o.engine.checkStatus(o.engine.funcs.setIntraOpNumThreads(o.handle, num))
}

// SetInterOpNumThreads 设置算子间并行线程数。
func (o *SessionOptions) SetInterOpNumThreads(num int32) error {
	if o.engine.funcs.setInterOpNumThreads == nil {
		return errAPIUnavailable(idxSetInterOpNumThreads)
	}
	return o.engine.checkStatus(o.engine.funcs.setInterOpNumThreads(o.handle, num))
}

// SetExecutionMode 设置执行模式（顺序 / 并行）。
func (o *SessionOptions) SetExecutionMode(mode ExecutionMode) error {
	if o.engine.funcs.setSessionExecutionMode == nil {
		return errAPIUnavailable(idxSetSessionExecutionMode)
	}
	return o.engine.checkStatus(o.engine.funcs.setSessionExecutionMode(o.handle, int32(mode)))
}

// SetGraphOptimizationLevel 设置图优化级别。
func (o *SessionOptions) SetGraphOptimizationLevel(level GraphOptimizationLevel) error {
	if o.engine.funcs.setSessionGraphOptimizationLevel == nil {
		return errAPIUnavailable(idxSetSessionGraphOptimizationLevel)
	}
	return o.engine.checkStatus(o.engine.funcs.setSessionGraphOptimizationLevel(o.handle, int32(level)))
}

// SetLogSeverityLevel 设置会话日志级别。
func (o *SessionOptions) SetLogSeverityLevel(level LoggingLevel) error {
	if o.engine.funcs.setSessionLogSeverityLevel == nil {
		return errAPIUnavailable(idxSetSessionLogSeverityLevel)
	}
	return o.engine.checkStatus(o.engine.funcs.setSessionLogSeverityLevel(o.handle, int32(level)))
}

// SetCpuMemArena 设置 CPU 内存池策略：
//
//	false: 禁用内存池，推理速度稍慢，但 Destroy 后立即归还内存给 OS
//	true:  启用内存池，推理速度最快，但 Destroy 后内存会被缓存以供复用（默认）
func (o *SessionOptions) SetCpuMemArena(useArena bool) error {
	if useArena {
		return o.engine.checkStatus(o.engine.funcs.enableCpuMemArena(o.handle))
	}
	return o.engine.checkStatus(o.engine.funcs.disableCpuMemArena(o.handle))
}

// AddFreeDimensionOverrideByName 按维度名固化模型的动态维度。
//
// 动态模型（如 YOLO）的输入维度常以 -1 出现，在创建 Session 前按维度名覆盖即可消除动态维：
//
//	opts, _ := engine.NewSessionOptions()
//	opts.AddFreeDimensionOverrideByName("batch", 1)
//	session, _ := engine.NewSession("yolo11n.onnx", opts)
func (o *SessionOptions) AddFreeDimensionOverrideByName(name string, dim int64) error {
	return o.setFreeDimensionOverride(name, dim, o.engine.funcs.addFreeDimensionOverrideByName)
}

// AddFreeDimensionOverride 按维度符号（denotation）固化动态维度。
func (o *SessionOptions) AddFreeDimensionOverride(denotation string, dim int64) error {
	return o.setFreeDimensionOverride(denotation, dim, o.engine.funcs.addFreeDimensionOverride)
}

func (o *SessionOptions) setFreeDimensionOverride(key string, dim int64, fn func(SessionOptionsHandle, *byte, int64) StatusHandle) error {
	if fn == nil {
		return errAPIUnavailable(idxAddFreeDimensionOverride)
	}
	keyPtr, err := stringToCString(key)
	if err != nil {
		return err
	}
	return o.engine.checkStatus(fn(o.handle, keyPtr, dim))
}

// Destroy 释放会话配置。
func (o *SessionOptions) Destroy() {
	if o.handle != 0 {
		o.engine.funcs.releaseSessionOptions(o.handle)
		o.handle = 0
	}
}
