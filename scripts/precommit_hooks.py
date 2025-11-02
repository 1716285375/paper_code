#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pre-commit hooks 辅助脚本
用于在本地配置中执行各种检查
"""
import sys
import pathlib
import re
import json
import yaml


def check_trailing_whitespace():
    """检查尾随空格"""
    files = [f for f in sys.argv[1:] if pathlib.Path(f).is_file()]
    for f in files:
        try:
            content = pathlib.Path(f).read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(content.splitlines(), 1):
                if line and line[-1] in " \t":
                    print(f"{f}:{i}: Trailing whitespace found")
                    sys.exit(1)
        except Exception:
            pass


def fix_end_of_file():
    """修复文件末尾换行符"""
    for f in sys.argv[1:]:
        try:
            p = pathlib.Path(f)
            if p.is_file() and p.stat().st_size > 0:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if not p.read_bytes().endswith(b"\n"):
                    p.write_text(content.rstrip() + "\n", encoding="utf-8")
        except Exception:
            pass


def check_large_files():
    """检查大文件"""
    max_size = 1048576  # 1MB
    for f in sys.argv[1:]:
        try:
            p = pathlib.Path(f)
            if p.is_file():
                size = p.stat().st_size
                if size > max_size:
                    print(f"{f}: {size} bytes (>{max_size} bytes)")
                    sys.exit(1)
        except Exception:
            pass


def check_merge_conflict():
    """检查合并冲突标记"""
    conflicts = ["<<<<<<<", "=======", ">>>>>>>"]
    for f in sys.argv[1:]:
        try:
            p = pathlib.Path(f)
            if p.is_file():
                content = p.read_text(encoding="utf-8", errors="ignore")
                if any(marker in content for marker in conflicts):
                    print(f"{f}: Merge conflict markers found")
                    sys.exit(1)
        except Exception:
            pass


def check_debug_statements():
    """检查调试语句"""
    patterns = [r"\bpdb\b", r"\bipdb\b", r"\bpudb\b", r"breakpoint\(\)"]
    for f in sys.argv[1:]:
        try:
            p = pathlib.Path(f)
            if p.is_file() and p.suffix == ".py":
                content = p.read_text(encoding="utf-8", errors="ignore")
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        print(f"{f}: Debug statement found")
                        sys.exit(1)
        except Exception:
            pass


def check_yaml():
    """检查YAML文件格式"""
    for f in sys.argv[1:]:
        try:
            p = pathlib.Path(f)
            if p.is_file():
                yaml.safe_load(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"{f}: YAML error - {e}")
            sys.exit(1)


def check_json():
    """检查JSON文件格式"""
    for f in sys.argv[1:]:
        try:
            p = pathlib.Path(f)
            if p.is_file():
                json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"{f}: JSON error - {e}")
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/precommit_hooks.py <hook-name> [files...]")
        sys.exit(1)
    
    hook_name = sys.argv[1]
    
    if hook_name == "trailing-whitespace":
        check_trailing_whitespace()
    elif hook_name == "end-of-file-fixer":
        fix_end_of_file()
    elif hook_name == "large-files":
        check_large_files()
    elif hook_name == "merge-conflict":
        check_merge_conflict()
    elif hook_name == "debug-statements":
        check_debug_statements()
    elif hook_name == "check-yaml":
        check_yaml()
    elif hook_name == "check-json":
        check_json()
    else:
        print(f"Unknown hook: {hook_name}")
        sys.exit(1)

