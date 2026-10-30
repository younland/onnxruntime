"""依据 onnxruntime 官方头文件生成 / 校验 api.go。

用法:
    python scripts/gen_api_go.py

流程:
    1. 从 GitHub 下载 v1.23.2 / v1.24.4 / v1.25.1 / v1.26.0 / v1.27.1 的 onnxruntime_c_api.h；
    2. 解析每个版本 struct OrtApi 的字段列表，校验“只追加”不变量
       （旧版本字段列表必须是新版本的严格前缀）；
    3. 用 1.27 的字段列表重新生成 api.go（ortApi 结构体 + apiLastIndex 边界表）。

verify-only 模式（不覆盖 api.go）:
    python scripts/gen_api_go.py --check
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
API_GO = REPO_ROOT / "api.go"
TMP = REPO_ROOT / ".tmp_headers"
VERSIONS = ["1.23.2", "1.24.4", "1.25.1", "1.26.0", "1.27.1",
            "1.28.3", "1.29.1", "1.30.1", "1.31.1"]

FIELD_RE = re.compile(r"\(\s*ORT_API_CALL\*\s*(\w+)\s*\)")
MACRO_RE = re.compile(r"ORT_API2_STATUS\(\s*(\w+)")
CLASS_RELEASE_RE = re.compile(r"ORT_CLASS_RELEASE\(\s*(\w+)\s*\)")


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def extract_struct(text: str) -> str:
    m = re.search(r"struct\s+OrtApi\s*\{", text)
    if not m:
        raise RuntimeError("struct OrtApi not found")
    start = m.end()
    end = text.index("\n};", start)
    return text[start:end]


def parse_fields(header_path: Path) -> list:
    text = strip_comments(header_path.read_text(encoding="utf-8"))
    body = extract_struct(text)
    fields = []
    for chunk in body.split(";"):
        m = FIELD_RE.search(chunk) or MACRO_RE.search(chunk)
        if m:
            fields.append(m.group(1))
            continue
        m = CLASS_RELEASE_RE.search(chunk)
        if m:
            fields.append("Release" + m.group(1))
    return fields


def download_headers() -> None:
    TMP.mkdir(exist_ok=True)
    for v in VERSIONS:
        out = TMP / f"onnxruntime_c_api_{v}.h"
        if out.exists() and out.stat().st_size > 1000:
            continue
        url = (
            f"https://raw.githubusercontent.com/microsoft/onnxruntime/"
            f"v{v}/include/onnxruntime/core/session/onnxruntime_c_api.h"
        )
        print(f"downloading {url}")
        data = urllib.request.urlopen(url, timeout=120).read()
        out.write_bytes(data)


def build_struct_text(fields: list, bounds: dict) -> str:
    lines = []
    lines.append("package ort")
    lines.append("")
    lines.append("import \"unsafe\"")
    lines.append("")
    lines.append("// 本文件由 scripts/gen_api_go.py 依据 onnxruntime 官方头文件生成，请勿手工修改。")
    lines.append("// 字段顺序与索引已经过 v1.23.2 ~ v1.31.1 九个版本头文件的逐一解析与前缀校验。")
    lines.append("")
    lines.append("// ortApiBase 对应 C 侧 OrtApiBase。")
    lines.append("type ortApiBase struct {")
    lines.append("\tGetAPI           uintptr")
    lines.append("\tGetVersionString uintptr")
    lines.append("}")
    lines.append("")
    lines.append("// ortApi 镜像 C 侧 struct OrtApi（只追加的函数指针表，当前到 API 31）。")
    lines.append("// 低版本运行时下 GetAPI 返回更短的表：只允许访问索引 <= apiLastIndex[minor] 的字段。")
    lines.append("type ortApi struct {")
    for i, name in enumerate(fields):
        lines.append(f"\t{name} uintptr // {i}")
    lines.append("}")
    lines.append("")
    lines.append("// apiLastIndex 记录各 ORT minor 版本 OrtApi 表的最后一个有效索引（含）。")
    lines.append("// 数据来源：v1.23.2 ~ v1.31.1 各版本官方头文件逐一解析。")
    lines.append("var apiLastIndex = map[uint32]uint32{")
    for v in VERSIONS:
        minor = v.split(".")[1]
        lines.append(f"\t{minor}: {bounds[minor]},")
    lines.append("}")
    lines.append("")
    lines.append(f"// apiMaxVersion 是本绑定编译期支持的最高 API 版本（ORT {VERSIONS[-1].rsplit('.', 1)[0]}）。")
    lines.append(f"const apiMaxVersion = {VERSIONS[-1].split('.')[1]}")
    lines.append("")
    lines.append("var _ = unsafe.Sizeof(ortApi{})")
    lines.append("")
    return "\n".join(lines)


def gofmt(text: str) -> str:
    import shutil
    import subprocess

    if not shutil.which("gofmt"):
        return text
    proc = subprocess.run(
        ["gofmt"], input=text.encode("utf-8"), stdout=subprocess.PIPE, check=True
    )
    return proc.stdout.decode("utf-8")


def main() -> int:
    check_only = "--check" in sys.argv
    download_headers()

    all_fields = {}
    for v in VERSIONS:
        fields = parse_fields(TMP / f"onnxruntime_c_api_{v}.h")
        all_fields[v] = fields
        print(f"{v}: {len(fields)} fields, last = {fields[-1]}")

    latest = all_fields[VERSIONS[-1]]
    for v in VERSIONS[:-1]:
        fs = all_fields[v]
        if latest[: len(fs)] != fs:
            raise RuntimeError(f"{v} is not a prefix of {VERSIONS[-1]} -- append-only invariant broken")
    print("append-only prefix invariant OK")

    bounds = {v.split(".")[1]: len(fs) - 1 for v, fs in all_fields.items()}
    print("apiLastIndex:", json.dumps(bounds))

    text = gofmt(build_struct_text(latest, bounds))
    if check_only:
        if API_GO.read_text(encoding="utf-8") != text:
            print("api.go is out of date, rerun without --check")
            return 1
        print("api.go is up to date")
        return 0

    API_GO.write_text(text, encoding="utf-8")
    print(f"wrote {API_GO} with {len(latest)} fields")
    return 0


if __name__ == "__main__":
    sys.exit(main())
