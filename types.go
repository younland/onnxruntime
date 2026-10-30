package ort

// 句柄与枚举类型定义，对应 onnxruntime C API 的 opaque 指针与 enum。
type (
	StatusHandle                    uintptr
	EnvHandle                       uintptr
	SessionHandle                   uintptr
	SessionOptionsHandle            uintptr
	ValueHandle                     uintptr
	AllocatorHandle                 uintptr
	MemoryInfoHandle                uintptr
	TensorTypeAndShapeInfoHandle    uintptr
	RunOptionsHandle                uintptr
	TypeInfoHandle                  uintptr
	CUDAProviderOptionsV2Handle     uintptr
	TensorRTProviderOptionsV2Handle uintptr

	ErrorCode             int32
	LoggingLevel          int32
	OnnxType              int32
	TensorElementDataType int32
	AllocatorType         int32
	MemType               int32
)

// TensorElementDataType ONNX Tensor 元素类型。
const (
	TensorElementDataTypeUndefined      TensorElementDataType = 0
	TensorElementDataTypeFloat          TensorElementDataType = 1
	TensorElementDataTypeUint8          TensorElementDataType = 2
	TensorElementDataTypeInt8           TensorElementDataType = 3
	TensorElementDataTypeUint16         TensorElementDataType = 4
	TensorElementDataTypeInt16          TensorElementDataType = 5
	TensorElementDataTypeInt32          TensorElementDataType = 6
	TensorElementDataTypeInt64          TensorElementDataType = 7
	TensorElementDataTypeString         TensorElementDataType = 8
	TensorElementDataTypeBool           TensorElementDataType = 9
	TensorElementDataTypeFloat16        TensorElementDataType = 10
	TensorElementDataTypeDouble         TensorElementDataType = 11
	TensorElementDataTypeUint32         TensorElementDataType = 12
	TensorElementDataTypeUint64         TensorElementDataType = 13
	TensorElementDataTypeComplex64      TensorElementDataType = 14
	TensorElementDataTypeComplex128     TensorElementDataType = 15
	TensorElementDataTypeBFloat16       TensorElementDataType = 16
	TensorElementDataTypeFloat8E4M3FN   TensorElementDataType = 17
	TensorElementDataTypeFloat8E4M3FNUZ TensorElementDataType = 18
	TensorElementDataTypeFloat8E5M2     TensorElementDataType = 19
	TensorElementDataTypeFloat8E5M2FNUZ TensorElementDataType = 20
	TensorElementDataTypeUint4          TensorElementDataType = 21
	TensorElementDataTypeInt4           TensorElementDataType = 22
	TensorElementDataTypeFloat4E2M1     TensorElementDataType = 23
)

const (
	DeviceAllocator AllocatorType = 0
	ArenaAllocator  AllocatorType = 1

	DefaultMemType MemType = 0
)

// LoggingLevel 日志级别。
const (
	LogVerbose LoggingLevel = 0
	LogInfo    LoggingLevel = 1
	LogWarning LoggingLevel = 2
	LogError   LoggingLevel = 3
	LogFatal   LoggingLevel = 4
)

// OnnxType Value 的类别。
const (
	OnnxTypeUnknown      OnnxType = 0
	OnnxTypeTensor       OnnxType = 1
	OnnxTypeSequence     OnnxType = 2
	OnnxTypeMap          OnnxType = 3
	OnnxTypeOpaque       OnnxType = 4
	OnnxTypeSparseTensor OnnxType = 5
	OnnxTypeOptional     OnnxType = 6
)

// ErrorCode OrtStatus 错误码。
const (
	ErrOK                             ErrorCode = 0
	ErrFail                           ErrorCode = 1
	ErrInvalidArgument                ErrorCode = 2
	ErrNoSuchFile                     ErrorCode = 3
	ErrNoModelLoaded                  ErrorCode = 4
	ErrEngineError                    ErrorCode = 5
	ErrRuntimeException               ErrorCode = 6
	ErrInvalidProtobuf                ErrorCode = 7
	ErrModelLoaded                    ErrorCode = 8
	ErrNotImplemented                 ErrorCode = 9
	ErrInvalidFileFormat              ErrorCode = 10
	ErrFileExists                     ErrorCode = 11
	ErrPermissionDenied               ErrorCode = 12
	ErrRuntimeOptimizationUnavailable ErrorCode = 13
	ErrExternalDataNotAccessible      ErrorCode = 15
)

// GraphOptimizationLevel 图优化级别（SessionOptions.SetGraphOptimizationLevel）。
type GraphOptimizationLevel int32

const (
	GraphOptimizationLevelNone      GraphOptimizationLevel = 0
	GraphOptimizationLevelBasic     GraphOptimizationLevel = 1
	GraphOptimizationLevelExtended  GraphOptimizationLevel = 2
	GraphOptimizationLevelAll       GraphOptimizationLevel = 99
	GraphOptimizationLevelEnableAll GraphOptimizationLevel = 99
)

// ExecutionMode 会话执行模式（SessionOptions.SetExecutionMode）。
type ExecutionMode int32

const (
	ExecutionModeSequential ExecutionMode = 0
	ExecutionModeParallel   ExecutionMode = 1
)
