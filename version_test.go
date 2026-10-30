package ort

import "testing"

func TestParseVersion(t *testing.T) {
	cases := []struct {
		in      string
		want    Version
		wantErr bool
	}{
		{"1.23.2", Version{1, 23, 2}, false},
		{"1.23", Version{1, 23, 0}, false},
		{"1.27.0-dev", Version{1, 27, 0}, false},
		{"1.26.0+gpu", Version{1, 26, 0}, false},
		{" 1.23.2 ", Version{1, 23, 2}, false},
		{"garbage", Version{}, true},
		{"1", Version{}, true},
		{"1.x.2", Version{}, true},
		{"", Version{}, true},
	}
	for _, c := range cases {
		got, err := parseVersion(c.in)
		if c.wantErr {
			if err == nil {
				t.Errorf("parseVersion(%q) expected error, got %v", c.in, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseVersion(%q) unexpected error: %v", c.in, err)
			continue
		}
		if got != c.want {
			t.Errorf("parseVersion(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestCheckVersionSupport(t *testing.T) {
	cases := []struct {
		name     string
		v        Version
		strict   bool
		wantErr  bool
		wantWarn bool
	}{
		{"1.22.9 refused", Version{1, 22, 9}, false, true, false},
		{"1.0.0 refused", Version{1, 0, 0}, false, true, false},
		{"1.23.0 ok", Version{1, 23, 0}, false, false, false},
		{"1.23.2 ok", Version{1, 23, 2}, false, false, false},
		{"1.25.1 ok", Version{1, 25, 1}, false, false, false},
		{"1.27.0 ok", Version{1, 27, 0}, false, false, false},
		{"1.27.9 ok", Version{1, 27, 9}, false, false, false},
		{"1.28.0 ok", Version{1, 28, 0}, false, false, false},
		{"1.31.1 ok", Version{1, 31, 1}, false, false, false},
		{"1.32.0 warn", Version{1, 32, 0}, false, false, true},
		{"1.32.0 strict refused", Version{1, 32, 0}, true, true, false},
		{"2.0.0 warn", Version{2, 0, 0}, false, false, true},
		{"2.0.0 strict refused", Version{2, 0, 0}, true, true, false},
	}
	for _, c := range cases {
		warn, err := checkVersionSupport(c.v, c.strict)
		if c.wantErr && err == nil {
			t.Errorf("%s: expected error, got nil (warn=%q)", c.name, warn)
		}
		if !c.wantErr && err != nil {
			t.Errorf("%s: unexpected error: %v", c.name, err)
		}
		if c.wantWarn && warn == "" {
			t.Errorf("%s: expected warning, got none", c.name)
		}
		if !c.wantWarn && warn != "" {
			t.Errorf("%s: unexpected warning %q", c.name, warn)
		}
	}
}

func TestResolveAPIVersion(t *testing.T) {
	cases := []struct {
		v    Version
		want uint32
	}{
		{Version{1, 23, 0}, 23},
		{Version{1, 24, 4}, 24},
		{Version{1, 31, 1}, 31},
		{Version{1, 32, 0}, apiMaxVersion},
		{Version{1, 99, 0}, apiMaxVersion},
		{Version{2, 0, 0}, apiMaxVersion},
	}
	for _, c := range cases {
		if got := resolveAPIVersion(c.v); got != c.want {
			t.Errorf("resolveAPIVersion(%v) = %d, want %d", c.v, got, c.want)
		}
	}
}

func TestAPILastIndexBounds(t *testing.T) {
	want := map[uint32]uint32{
		23: 381, 24: 404, 25: 408, 26: 408, 27: 411,
		28: 412, 29: 413, 30: 414, 31: 416,
	}
	if len(apiLastIndex) != len(want) {
		t.Fatalf("apiLastIndex has %d entries, want %d", len(apiLastIndex), len(want))
	}
	prev := uint32(0)
	for minor := uint32(23); minor <= apiMaxVersion; minor++ {
		got, ok := apiLastIndex[minor]
		if !ok {
			t.Fatalf("apiLastIndex missing minor %d", minor)
		}
		if want[minor] != got {
			t.Errorf("apiLastIndex[%d] = %d, want %d (regenerate api.go via scripts/gen_api_go.py)", minor, got, want[minor])
		}
		if got < prev {
			t.Errorf("apiLastIndex not monotonic at minor %d: %d < %d", minor, got, prev)
		}
		prev = got
	}
}
