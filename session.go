package ort

import (
	"fmt"
	"unsafe"
)

// TensorInfo 描述一个输入 / 输出张量的元信息。
type TensorInfo struct {
	Name     string
	DataType TensorElementDataType
	Shape    []int64
}

// ElementCount 返回 Shape 各维度的乘积。含动态维度（-1）或溢出时第二个返回值为 false。
func (t TensorInfo) ElementCount() (int64, bool) {
	return shapeElementCount(t.Shape)
}

// Session 推理会话。
type Session struct {
	handle  SessionHandle
	engine  *Engine
	Inputs  []TensorInfo
	Outputs []TensorInfo

	// modelData 持有 NewSessionFromBytes 传入的模型字节，保证 Session 存活期间缓冲区不被回收
	modelData []byte
}

// NewSession 从模型文件创建会话。
func (e *Engine) NewSession(modelPath string, opts *SessionOptions) (*Session, error) {
	pathPtr, err := stringToPathPtr(modelPath)
	if err != nil {
		return nil, err
	}
	return e.newSession(opts, func(optHandle SessionOptionsHandle, h *SessionHandle) StatusHandle {
		return e.funcs.createSession(e.envHandle, pathPtr, optHandle, h)
	})
}

// NewSessionFromBytes 从内存中的模型字节创建会话，适用于 go:embed 等无文件场景。
func (e *Engine) NewSessionFromBytes(data []byte, opts *SessionOptions) (*Session, error) {
	if len(data) == 0 {
		return nil, fmt.Errorf("NewSessionFromBytes: model data is empty")
	}
	s, err := e.newSession(opts, func(optHandle SessionOptionsHandle, h *SessionHandle) StatusHandle {
		return e.funcs.createSessionFromArray(e.envHandle, unsafe.Pointer(slicePtr(data)), uintptr(len(data)), optHandle, h)
	})
	if err != nil {
		return nil, err
	}
	s.modelData = data
	return s, nil
}

func (e *Engine) newSession(opts *SessionOptions, create func(SessionOptionsHandle, *SessionHandle) StatusHandle) (*Session, error) {
	var optHandle SessionOptionsHandle
	if opts != nil {
		optHandle = opts.handle
	}

	var h SessionHandle
	if err := e.checkStatus(create(optHandle, &h)); err != nil {
		return nil, err
	}

	s := &Session{handle: h, engine: e}
	if err := s.initMetadata(); err != nil {
		s.Destroy()
		return nil, err
	}
	return s, nil
}

func (s *Session) initMetadata() error {
	inputs, err := s.collectMetadata(true)
	if err != nil {
		return err
	}
	outputs, err := s.collectMetadata(false)
	if err != nil {
		return err
	}
	s.Inputs, s.Outputs = inputs, outputs
	return nil
}

// collectMetadata 收集模型输入 / 输出元信息。isInput=true 收集输入，false 收集输出。
func (s *Session) collectMetadata(isInput bool) ([]TensorInfo, error) {
	f := s.engine.funcs
	count, name, typeInfo := f.sessionGetInputCount, f.sessionGetInputName, f.sessionGetInputTypeInfo
	if !isInput {
		count, name, typeInfo = f.sessionGetOutputCount, f.sessionGetOutputName, f.sessionGetOutputTypeInfo
	}

	var n uintptr
	if err := s.engine.checkStatus(count(s.handle, &n)); err != nil {
		return nil, err
	}

	var allocator AllocatorHandle
	if err := s.engine.checkStatus(f.getAllocatorWithDefaultOptions(&allocator)); err != nil {
		return nil, err
	}

	infos := make([]TensorInfo, n)
	for i := range infos {
		var namePtr *byte
		if err := s.engine.checkStatus(name(s.handle, uintptr(i), allocator, &namePtr)); err != nil {
			return nil, err
		}
		infos[i].Name = cStringToString(namePtr)
		f.allocatorFree(allocator, unsafe.Pointer(namePtr))

		var ti TypeInfoHandle
		if err := s.engine.checkStatus(typeInfo(s.handle, uintptr(i), &ti)); err != nil {
			return nil, err
		}
		dataType, shape, err := s.readTensorType(ti)
		f.releaseTypeInfo(ti)
		if err != nil {
			return nil, err
		}
		infos[i].DataType, infos[i].Shape = dataType, shape
	}
	return infos, nil
}

// readTensorType 从 OrtTypeInfo 解析出元素类型与形状。
func (s *Session) readTensorType(ti TypeInfoHandle) (TensorElementDataType, []int64, error) {
	f := s.engine.funcs

	var info TensorTypeAndShapeInfoHandle
	if err := s.engine.checkStatus(f.castTypeInfoToTensorInfo(ti, &info)); err != nil {
		return TensorElementDataTypeUndefined, nil, err
	}
	if info == 0 {
		return TensorElementDataTypeUndefined, nil, nil
	}

	var dataType TensorElementDataType
	if err := s.engine.checkStatus(f.getTensorElementType(info, &dataType)); err != nil {
		return TensorElementDataTypeUndefined, nil, err
	}

	var dimCount uintptr
	if err := s.engine.checkStatus(f.getDimensionsCount(info, &dimCount)); err != nil {
		return TensorElementDataTypeUndefined, nil, err
	}

	shape := make([]int64, dimCount)
	if dimCount > 0 {
		if err := s.engine.checkStatus(f.getDimensions(info, &shape[0], dimCount)); err != nil {
			return TensorElementDataTypeUndefined, nil, err
		}
	}
	return dataType, shape, nil
}

// Destroy 释放会话。必须先于 Engine.Destroy 调用。
func (s *Session) Destroy() {
	if s.handle != 0 {
		s.engine.funcs.releaseSession(s.handle)
		s.handle = 0
	}
}

// Run 执行推理。inputs 的 key 为输入名（见 Session.Inputs），返回的 Value 由调用方负责 Destroy。
func (s *Session) Run(inputs map[string]*Value) (map[string]*Value, error) {
	inputCount := len(inputs)
	outputCount := len(s.Outputs)

	if inputCount == 0 {
		return nil, fmt.Errorf("session.Run: no inputs provided, expect %v", tensorNames(s.Inputs))
	}

	// input
	inputNamePtrs := make([]unsafe.Pointer, inputCount)
	inputHandles := make([]ValueHandle, inputCount)
	i := 0
	for name, val := range inputs {
		if val == nil {
			return nil, fmt.Errorf("session.Run: input %q is nil", name)
		}
		if val.engine != s.engine {
			return nil, fmt.Errorf("session.Run: input %q was created by a different engine", name)
		}
		cName, err := stringToCString(name)
		if err != nil {
			return nil, err
		}
		inputNamePtrs[i] = unsafe.Pointer(cName)
		inputHandles[i] = val.handle
		i++
	}

	// output
	outputNamePtrs := make([]unsafe.Pointer, outputCount)
	outputHandles := make([]ValueHandle, outputCount)
	for i, out := range s.Outputs {
		cName, err := stringToCString(out.Name)
		if err != nil {
			return nil, err
		}
		outputNamePtrs[i] = unsafe.Pointer(cName)
	}

	status := s.engine.funcs.run(
		s.handle,
		0,
		slicePtr(inputNamePtrs),
		slicePtr(inputHandles),
		uintptr(inputCount),
		slicePtr(outputNamePtrs),
		uintptr(outputCount),
		slicePtr(outputHandles),
	)
	if err := s.engine.checkStatus(status); err != nil {
		return nil, fmt.Errorf("failed to run session: %w", err)
	}

	results := make(map[string]*Value, outputCount)
	for i := 0; i < outputCount; i++ {
		results[s.Outputs[i].Name] = &Value{
			handle: outputHandles[i],
			engine: s.engine,
		}
	}
	return results, nil
}

// tensorNames 提取元信息中的名字。
func tensorNames(infos []TensorInfo) []string {
	names := make([]string, len(infos))
	for i, info := range infos {
		names[i] = info.Name
	}
	return names
}
