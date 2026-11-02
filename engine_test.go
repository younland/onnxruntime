package ort

import "testing"

// TestEngineLifecycleAndVersion 引擎生命周期：加载 -> 版本协商 -> 释放。
func TestEngineLifecycleAndVersion(t *testing.T) {
	lib := findTestLibrary(t)

	engine, err := NewEngine(WithLibraryPath(lib))
	if err != nil {
		t.Fatalf("NewEngine(%s) failed: %v", lib, err)
	}
	defer engine.Destroy()

	v := engine.Version()
	if v.Major != 1 || v.Minor < minSupportedMinor || v.Minor > maxVerifiedMinor {
		t.Fatalf("unexpected runtime version %s from library %s", v, lib)
	}
	if engine.APIVersion() < minSupportedMinor || engine.APIVersion() > apiMaxVersion {
		t.Fatalf("unexpected api version %d", engine.APIVersion())
	}
	t.Logf("library=%s version=%s api=%d providers=%v", lib, engine.VersionString(), engine.APIVersion(), engine.GetAvailableProviders())
}

// TestNewEngineInvalidPath 加载不存在的动态库必须返回错误，不得 panic。
func TestNewEngineInvalidPath(t *testing.T) {
	if _, err := NewEngine(WithLibraryPath("nonexistent/onnxruntime.dll")); err == nil {
		t.Fatal("NewEngine with invalid library path must return an error")
	}
}

// TestSessionOptionsAddConfigEntry 配置项写入：空闲自旋开关等 session.* 键必须可设置。
func TestSessionOptionsAddConfigEntry(t *testing.T) {
	lib := findTestLibrary(t)

	engine, err := NewEngine(WithLibraryPath(lib))
	if err != nil {
		t.Fatalf("NewEngine(%s) failed: %v", lib, err)
	}
	defer engine.Destroy()

	opts, err := engine.NewSessionOptions()
	if err != nil {
		t.Fatalf("NewSessionOptions failed: %v", err)
	}
	defer opts.Destroy()

	if err := opts.AddConfigEntry("session.intra_op.allow_spinning", "0"); err != nil {
		t.Fatalf("AddConfigEntry(session.intra_op.allow_spinning) failed: %v", err)
	}
	if err := opts.AddConfigEntry("session.inter_op.allow_spinning", "0"); err != nil {
		t.Fatalf("AddConfigEntry(session.inter_op.allow_spinning) failed: %v", err)
	}
}
