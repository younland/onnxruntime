package ort

import (
	"fmt"
	"math"
	"unsafe"
)

// Number 是 GetTensorData 支持的数值类型集合。
type Number interface {
	~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// Value 是 ONNX Runtime 的 OrtValue 包装，当前仅支持 Tensor。
type Value struct {
	handle       ValueHandle
	engine       *Engine
	shape        []int64
	elementCount int

	// data 持有传给 CreateTensorWithDataAsOrtValue 的底层数据引用。
	//
	// onnxruntime 只保存指针、不拷贝数据，若不在此保留引用，NewTensor 返回后
	// 该缓冲区即不可达，GC 会回收并复用，而 ORT 仍按原指针读取。
	data any
}

// NewTensor 初始化 Tensor（使用 NewEngine 设置的默认引擎）。
//
// # Params:
//
//	shape: 形状
//	data: 数据，支持 []float32 / []float64 / []int64 / []int32 / []int16 / []int8 /
//	      []uint64 / []uint32 / []uint16 / []uint8 / []bool
//
// onnxruntime 不会拷贝 data，只持有其指针，数据有效性由返回的 Value 托管。
// 调用 Value.Destroy() 后，由 GetTensorData 取到的切片随之失效。
func NewTensor(shape []int64, data any) (*Value, error) {
	engine := getDefaultEngine()
	if engine == nil {
		return nil, fmt.Errorf("engine not initialized")
	}
	return engine.NewTensor(shape, data)
}

// NewTensor 在指定引擎上初始化 Tensor。
func (e *Engine) NewTensor(shape []int64, data any) (*Value, error) {
	dataType, typeSize, dataLen, dataPtr, err := parseInputData(data)
	if err != nil {
		return nil, err
	}

	// 元素个数由 shape 决定，onnxruntime 并不校验缓冲区长度，数据不足会越界读，这里提前拦截。
	// shape 含动态维度(-1)或乘积溢出时交由 onnxruntime 报错。
	if want, ok := shapeElementCount(shape); ok && int64(dataLen) < want {
		return nil, fmt.Errorf("data length %d is less than the %d elements required by shape %v", dataLen, want, shape)
	}

	var valHandle ValueHandle
	status := e.funcs.createTensorWithDataAsOrtValue(
		e.memInfo,
		dataPtr,
		uintptr(dataLen)*typeSize,
		slicePtr(shape),
		uintptr(len(shape)),
		dataType,
		&valHandle,
	)
	if err := e.checkStatus(status); err != nil {
		return nil, err
	}

	return &Value{handle: valHandle, engine: e, data: data}, nil
}

// shapeElementCount 计算 shape 的元素总数。
// 含负数（动态维度）或乘积溢出时返回 ok=false。
func shapeElementCount(shape []int64) (int64, bool) {
	total := int64(1)
	for _, dim := range shape {
		if dim < 0 || (dim != 0 && total > math.MaxInt64/dim) {
			return 0, false
		}
		total *= dim
	}
	return total, true
}

// GetShape 获取 Tensor 的维度信息。返回的是内部缓存切片本身，调用方不应修改其内容。
func (v *Value) GetShape() ([]int64, error) {
	if v.shape != nil {
		return v.shape, nil
	}

	info, err := v.getTypeAndShapeInfo()
	if err != nil {
		return nil, err
	}
	defer v.engine.funcs.releaseTensorTypeAndShapeInfo(info)

	var dimCount uintptr
	if err := v.engine.checkStatus(v.engine.funcs.getDimensionsCount(info, &dimCount)); err != nil {
		return nil, fmt.Errorf("failed to get dimensions count: %w", err)
	}

	// 标量张量的 dimCount 为 0，此时不能取 &v.shape[0]
	v.shape = make([]int64, dimCount)
	if dimCount > 0 {
		if err := v.engine.checkStatus(v.engine.funcs.getDimensions(info, &v.shape[0], dimCount)); err != nil {
			return nil, fmt.Errorf("failed to get dimensions: %w", err)
		}
	}
	return v.shape, nil
}

// GetElementCount 获取 Tensor 中的元素总数。
func (v *Value) GetElementCount() (int, error) {
	if v.elementCount > 0 {
		return v.elementCount, nil
	}

	info, err := v.getTypeAndShapeInfo()
	if err != nil {
		return 0, err
	}
	defer v.engine.funcs.releaseTensorTypeAndShapeInfo(info)

	var elementCount uintptr
	if err := v.engine.checkStatus(v.engine.funcs.getTensorShapeElementCount(info, &elementCount)); err != nil {
		return 0, fmt.Errorf("failed to get tensor shape element count: %w", err)
	}

	v.elementCount = int(elementCount)
	return v.elementCount, nil
}

// GetTensorData 获取 Tensor 数据，T 需与实际元素类型匹配。
func GetTensorData[T Number](v *Value) ([]T, error) {
	elementCount, err := v.GetElementCount()
	if err != nil {
		return nil, err
	}

	info, err := v.getTypeAndShapeInfo()
	if err != nil {
		return nil, fmt.Errorf("failed to get tensor type and shape info: %w", err)
	}
	defer v.engine.funcs.releaseTensorTypeAndShapeInfo(info)

	var dataType TensorElementDataType
	if err := v.engine.checkStatus(v.engine.funcs.getTensorElementType(info, &dataType)); err != nil {
		return nil, fmt.Errorf("failed to get tensor element type: %w", err)
	}

	var ptr unsafe.Pointer
	if err := v.engine.checkStatus(v.engine.funcs.getTensorMutableData(v.handle, &ptr)); err != nil {
		return nil, fmt.Errorf("failed to get tensor mutable data: %w", err)
	}

	var rawData any
	switch dataType {
	case TensorElementDataTypeFloat:
		rawData = unsafe.Slice((*float32)(ptr), elementCount)
	case TensorElementDataTypeDouble:
		rawData = unsafe.Slice((*float64)(ptr), elementCount)
	case TensorElementDataTypeInt64:
		rawData = unsafe.Slice((*int64)(ptr), elementCount)
	case TensorElementDataTypeInt32:
		rawData = unsafe.Slice((*int32)(ptr), elementCount)
	case TensorElementDataTypeInt16:
		rawData = unsafe.Slice((*int16)(ptr), elementCount)
	case TensorElementDataTypeInt8:
		rawData = unsafe.Slice((*int8)(ptr), elementCount)
	case TensorElementDataTypeUint64:
		rawData = unsafe.Slice((*uint64)(ptr), elementCount)
	case TensorElementDataTypeUint32:
		rawData = unsafe.Slice((*uint32)(ptr), elementCount)
	case TensorElementDataTypeUint16:
		rawData = unsafe.Slice((*uint16)(ptr), elementCount)
	case TensorElementDataTypeUint8:
		rawData = unsafe.Slice((*uint8)(ptr), elementCount)
	case TensorElementDataTypeBool:
		rawData = unsafe.Slice((*bool)(ptr), elementCount)
	default:
		return nil, fmt.Errorf("unsupported tensor element type: %d", dataType)
	}

	if data, ok := rawData.([]T); ok {
		return data, nil
	}

	var t T
	return nil, fmt.Errorf("tensor data type mismatch: actual ORT type %d does not match requested Go type %T", dataType, t)
}

func (v *Value) getTypeAndShapeInfo() (TensorTypeAndShapeInfoHandle, error) {
	var info TensorTypeAndShapeInfoHandle
	if err := v.engine.checkStatus(v.engine.funcs.getTensorTypeAndShape(v.handle, &info)); err != nil {
		return 0, err
	}
	return info, nil
}

// parseInputData 解析输入数据，返回 (元素类型, 单元素字节数, 元素个数, 数据指针)。
func parseInputData(data any) (TensorElementDataType, uintptr, int, unsafe.Pointer, error) {
	switch d := data.(type) {
	case []float32:
		return TensorElementDataTypeFloat, 4, len(d), unsafe.Pointer(slicePtr(d)), nil
	case []float64:
		return TensorElementDataTypeDouble, 8, len(d), unsafe.Pointer(slicePtr(d)), nil
	case []int64:
		return TensorElementDataTypeInt64, 8, len(d), unsafe.Pointer(slicePtr(d)), nil
	case []int32:
		return TensorElementDataTypeInt32, 4, len(d), unsafe.Pointer(slicePtr(d)), nil
	case []int16:
		return TensorElementDataTypeInt16, 2, len(d), unsafe.Pointer(slicePtr(d)), nil
	case []int8:
		return TensorElementDataTypeInt8, 1, len(d), unsafe.Pointer(slicePtr(d)), nil
	case []uint64:
		return TensorElementDataTypeUint64, 8, len(d), unsafe.Pointer(slicePtr(d)), nil
	case []uint32:
		return TensorElementDataTypeUint32, 4, len(d), unsafe.Pointer(slicePtr(d)), nil
	case []uint16:
		return TensorElementDataTypeUint16, 2, len(d), unsafe.Pointer(slicePtr(d)), nil
	case []uint8:
		return TensorElementDataTypeUint8, 1, len(d), unsafe.Pointer(slicePtr(d)), nil
	case []bool:
		return TensorElementDataTypeBool, 1, len(d), unsafe.Pointer(slicePtr(d)), nil
	default:
		return TensorElementDataTypeUndefined, 0, 0, nil, fmt.Errorf("unsupported input type: %T", data)
	}
}

// Destroy 释放 Value。
func (v *Value) Destroy() {
	if v.handle != 0 {
		v.engine.funcs.releaseValue(v.handle)
		v.handle = 0
	}
}
