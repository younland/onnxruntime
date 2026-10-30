package ort

import (
	"fmt"
	"strconv"
	"strings"
)

// 版本支持范围：最低 1.23，最高已验证 1.31。
// ORT 的 minor 版本号与 C API 版本一致（1.23 -> API 23，…，1.31 -> API 31）。
const (
	minSupportedMajor = 1
	minSupportedMinor = 23
	maxVerifiedMajor  = 1
	maxVerifiedMinor  = 31
)

// Version 表示解析后的 onnxruntime 版本。
type Version struct {
	Major uint32
	Minor uint32
	Patch uint32
}

func (v Version) String() string {
	return fmt.Sprintf("%d.%d.%d", v.Major, v.Minor, v.Patch)
}

// parseVersion 解析 GetVersionString 返回的版本字符串，例如 "1.23.2"。
//
// 容忍两段的 "1.23" 以及带后缀的 "1.23.0-dev" 形式；无法解析出 major/minor 时返回错误。
func parseVersion(s string) (Version, error) {
	s = strings.TrimSpace(s)
	if i := strings.IndexAny(s, "-+"); i >= 0 {
		s = s[:i]
	}
	parts := strings.Split(s, ".")
	if len(parts) < 2 {
		return Version{}, fmt.Errorf("invalid version string: %q", s)
	}
	var v Version
	nums := []*uint32{&v.Major, &v.Minor, &v.Patch}
	for i, p := range parts {
		if i >= len(nums) {
			break
		}
		n, err := strconv.ParseUint(strings.TrimSpace(p), 10, 32)
		if err != nil {
			return Version{}, fmt.Errorf("invalid version string: %q", s)
		}
		*nums[i] = uint32(n)
	}
	return v, nil
}

// checkVersionSupport 版本门禁：
//
//   - 低于 1.23：返回错误，拒绝启动；
//   - 1.23 ~ 1.31：正常；
//   - 高于 1.31（含 2.x）：strict 时返回错误拒绝启动，否则返回警告文案由调用方打印。
//
// 第二个返回值为警告信息，仅在 err 为 nil 时可能非空。
func checkVersionSupport(v Version, strict bool) (warn string, err error) {
	belowMin := v.Major < minSupportedMajor ||
		(v.Major == minSupportedMajor && v.Minor < minSupportedMinor)
	if belowMin {
		return "", fmt.Errorf(
			"onnxruntime %d.%d or higher is required, but the loaded library is %s (versions below 1.%d are not supported)",
			minSupportedMajor, minSupportedMinor, v, minSupportedMinor)
	}

	aboveMax := v.Major > maxVerifiedMajor ||
		(v.Major == maxVerifiedMajor && v.Minor > maxVerifiedMinor)
	if aboveMax {
		msg := fmt.Sprintf(
			"onnxruntime %s is newer than the highest verified version %d.%d; compatibility is unverified",
			v, maxVerifiedMajor, maxVerifiedMinor)
		if strict {
			return "", fmt.Errorf("%s (rejected by strict version mode)", msg)
		}
		return msg, nil
	}
	return "", nil
}

// resolveAPIVersion 由运行时版本推导可用的 OrtApi 表版本。
// 只增不减的函数表使得高版本表兼容低版本调用；高于已验证上限时按上限处理。
func resolveAPIVersion(v Version) uint32 {
	// 未来 2.x 版本无法假设 minor 与 API 版本一致，保守封顶在 apiMaxVersion
	if v.Major != 1 {
		return apiMaxVersion
	}
	if v.Minor > apiMaxVersion {
		return apiMaxVersion
	}
	return v.Minor
}
