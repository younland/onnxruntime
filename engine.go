package ort

import (
	"fmt"
	"log"
	"sync"

	"github.com/ebitengine/purego"
	"github.com/younland/onnxruntime/internal/sys"
)

// DefaultEnvName 是 CreateEnv 的默认 logid。
const DefaultEnvName = "younland/onnxruntime"

// defaultEngine 由 NewEngine 设置，供包级 NewTensor 使用。
var (
	defaultEngineMu sync.RWMutex
	defaultEngine   *Engine
)

func getDefaultEngine() *Engine {
	defaultEngineMu.RLock()
	defer defaultEngineMu.RUnlock()
	return defaultEngine
}

func setDefaultEngine(e *Engine) {
	defaultEngineMu.Lock()
	defer defaultEngineMu.Unlock()
	defaultEngine = e
}

// errAPIUnavailable 构造索引不可用错误。
func errAPIUnavailable(idx uint32) error {
	return fmt.Errorf("onnxruntime C API function at index %d is not available in this runtime", idx)
}

// Engine 推理引擎上下文。一套二进制可加载 1.23 ~ 1.31 任意版本的 onnxruntime 动态库。
type Engine struct {
	handle    uintptr
	version   Version // 运行时真实版本（来自 GetVersionString）
	apiVer    uint32  // 协商出的 OrtApi 表版本（<= apiMaxVersion）
	lastIndex uint32  // 当前 OrtApi 表的最后一个有效索引

	apiBase *ortApiBase
	api     *ortApi
	funcs   *apiFuncs

	envHandle EnvHandle
	memInfo   MemoryInfoHandle
}

// engineConfig NewEngine 的可选配置。
type engineConfig struct {
	libraryPath string
	strict      bool
	envName     string
}

// Option 引擎配置选项。
type Option func(*engineConfig)

// WithLibraryPath 指定 onnxruntime 动态库路径，默认 DefaultLibraryPath()。
func WithLibraryPath(path string) Option {
	return func(c *engineConfig) { c.libraryPath = path }
}

// WithStrictVersion 开启严格版本模式：加载高于已验证上限（1.31）的 onnxruntime 时
// 拒绝启动而不是仅警告。默认关闭。
func WithStrictVersion(strict bool) Option {
	return func(c *engineConfig) { c.strict = strict }
}

// WithEnvName 指定 CreateEnv 的 logid，默认 DefaultEnvName。
func WithEnvName(name string) Option {
	return func(c *engineConfig) { c.envName = name }
}

// NewEngine 加载 onnxruntime 动态库并完成初始化。
//
// 版本门禁：低于 1.23 拒绝启动；1.23 ~ 1.31 直接工作；高于 1.31 默认打印警告后放行，
// WithStrictVersion(true) 时拒绝。函数表按运行时实际版本协商，同一套代码可跨版本运行。
func NewEngine(options ...Option) (*Engine, error) {
	cfg := engineConfig{
		libraryPath: DefaultLibraryPath(),
		envName:     DefaultEnvName,
	}
	for _, o := range options {
		o(&cfg)
	}

	handle, err := sys.LoadLibrary(cfg.libraryPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load onnxruntime library: %w (请确认 CPU/GPU 构建与依赖环境匹配)", err)
	}
	fail := func(err error) (*Engine, error) {
		sys.FreeLibrary(handle)
		return nil, err
	}

	// OrtGetApiBase 是所有版本都有的导出函数
	var getAPIBase func() *ortApiBase
	purego.RegisterLibFunc(&getAPIBase, handle, "OrtGetApiBase")
	apiBase := getAPIBase()
	if apiBase == nil {
		return fail(fmt.Errorf("OrtGetApiBase returned nil"))
	}

	e := &Engine{handle: handle, apiBase: apiBase, funcs: &apiFuncs{}}

	purego.RegisterFunc(&e.funcs.getVersionString, apiBase.GetVersionString)
	ver, err := parseVersion(e.funcs.getVersionString())
	if err != nil {
		return fail(fmt.Errorf("failed to parse onnxruntime version: %w", err))
	}
	e.version = ver

	// 版本门禁
	warn, err := checkVersionSupport(ver, cfg.strict)
	if err != nil {
		return fail(err)
	}
	if warn != "" {
		log.Printf("[onnxruntime] WARNING: %s", warn)
	}

	// 协商 OrtApi 表版本并绑定函数
	e.apiVer = resolveAPIVersion(ver)
	e.lastIndex = apiLastIndex[e.apiVer]
	var getAPI func(uint32) *ortApi
	purego.RegisterFunc(&getAPI, apiBase.GetAPI)
	e.api = getAPI(e.apiVer)
	if e.api == nil {
		return fail(fmt.Errorf("failed to get OrtApi for version %d (onnxruntime %s)", e.apiVer, ver))
	}
	e.bindAPIFuncs()

	// 核心函数自检
	if err := e.require(
		idxCreateStatus, idxGetErrorCode, idxGetErrorMessage, idxReleaseStatus,
		idxCreateEnv, idxReleaseEnv,
		idxGetAllocatorWithDefaultOptions, idxAllocatorFree,
		idxCreateCpuMemoryInfo, idxReleaseMemoryInfo,
		idxCreateSessionOptions, idxReleaseSessionOptions,
		idxCreateSession, idxCreateSessionFromArray, idxRun, idxReleaseSession,
		idxSessionGetInputCount, idxSessionGetOutputCount,
		idxSessionGetInputName, idxSessionGetOutputName,
		idxSessionGetInputTypeInfo, idxSessionGetOutputTypeInfo,
		idxCastTypeInfoToTensorInfo, idxReleaseTypeInfo,
		idxCreateTensorWithDataAsOrtValue, idxGetTensorMutableData,
		idxGetTensorTypeAndShape, idxGetTensorElementType,
		idxGetDimensionsCount, idxGetDimensions, idxGetTensorShapeElementCount,
		idxReleaseValue, idxReleaseTensorTypeAndShapeInfo,
	); err != nil {
		return fail(fmt.Errorf("onnxruntime %s lacks required C API functions: %w", ver, err))
	}

	if err := e.initEnv(cfg.envName); err != nil {
		return fail(err)
	}
	if err := e.initMemInfo(); err != nil {
		e.cleanup()
		return fail(err)
	}

	setDefaultEngine(e)
	return e, nil
}

func (e *Engine) initEnv(name string) error {
	namePtr, err := stringToCString(name)
	if err != nil {
		return err
	}
	if err := e.checkStatus(e.funcs.createEnv(LogError, namePtr, &e.envHandle)); err != nil {
		return fmt.Errorf("failed to create env: %w", err)
	}
	return nil
}

func (e *Engine) initMemInfo() error {
	if err := e.checkStatus(e.funcs.createCpuMemoryInfo(DeviceAllocator, DefaultMemType, &e.memInfo)); err != nil {
		return fmt.Errorf("failed to create cpu memory info: %w", err)
	}
	return nil
}

// Version 返回加载的运行时版本，例如 "1.23.2"。
func (e *Engine) Version() Version { return e.version }

// VersionString 返回加载的运行时版本字符串，例如 "1.23.2"。
func (e *Engine) VersionString() string { return e.version.String() }

// APIVersion 返回与运行时协商出的 OrtApi 表版本（23 ~ 27）。
func (e *Engine) APIVersion() uint32 { return e.apiVer }

// cleanup 释放 env / memory info，可重复调用。
func (e *Engine) cleanup() {
	if e.memInfo != 0 {
		e.funcs.releaseMemoryInfo(e.memInfo)
		e.memInfo = 0
	}
	if e.envHandle != 0 {
		e.funcs.releaseEnv(e.envHandle)
		e.envHandle = 0
	}
}

// Destroy 释放资源。
//
// 调用顺序：session 由 env 创建并引用 env，必须先释放所有 Session（及由它们产出的
// Value），再调用本方法。Destroy 之后包级 NewTensor 会返回 "engine not initialized"。
//
// 注意：与官方 C# / Python 绑定一致，这里不卸载动态库本身，避免 ORT 静态析构崩溃。
func (e *Engine) Destroy() {
	e.cleanup()

	defaultEngineMu.Lock()
	if defaultEngine == e {
		defaultEngine = nil
	}
	defaultEngineMu.Unlock()

	e.handle = 0
}

// checkStatus 检查 OrtStatus 并转为 Go error。
func (e *Engine) checkStatus(status StatusHandle) error {
	if status == 0 {
		return nil
	}
	defer e.funcs.releaseStatus(status)

	code := e.funcs.getErrorCode(status)
	msgPtr := e.funcs.getErrorMessage(status)
	msg := cStringToString((*byte)(msgPtr))

	return fmt.Errorf("onnxruntime error [code %d]: %s", code, msg)
}
