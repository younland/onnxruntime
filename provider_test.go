package ort

import "testing"

// TestProvidersAvailability 不实际启用 GPU EP，只验证查询接口可用且 CPU 恒存在。
func TestProvidersAvailability(t *testing.T) {
	lib := findTestLibrary(t)

	engine, err := NewEngine(WithLibraryPath(lib))
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}
	defer engine.Destroy()

	if !engine.IsProviderAvailable(ProviderCPU) {
		t.Fatalf("CPUExecutionProvider must always be available, got %v", engine.GetAvailableProviders())
	}
	// CUDA / TensorRT 是否可用取决于加载的构建，这里仅做类型与接口冒烟。
	t.Logf("providers: %v", engine.GetAvailableProviders())
}

// TestEnableGPUReturnsErrorOnCPUOnlyBuild 验证在纯 CPU 构建上启用 GPU EP 会得到明确错误。
// 若加载的库实际包含 CUDA EP（GPU 构建），则跳过。
func TestEnableGPUReturnsErrorOnCPUOnlyBuild(t *testing.T) {
	lib := findTestLibrary(t)

	engine, err := NewEngine(WithLibraryPath(lib))
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}
	defer engine.Destroy()

	if engine.IsProviderAvailable(ProviderCUDA) || engine.IsProviderAvailable(ProviderTensorRT) {
		t.Skip("loaded library is a GPU build; negative-path test not applicable")
	}

	opts, err := engine.NewSessionOptions()
	if err != nil {
		t.Fatalf("NewSessionOptions failed: %v", err)
	}
	defer opts.Destroy()

	if err := opts.EnableCUDA(nil); err == nil {
		t.Fatal("EnableCUDA on a CPU-only build must return an error")
	} else {
		t.Logf("EnableCUDA error as expected: %v", err)
	}
	if err := opts.EnableTensorRT(nil); err == nil {
		t.Fatal("EnableTensorRT on a CPU-only build must return an error")
	} else {
		t.Logf("EnableTensorRT error as expected: %v", err)
	}
}
