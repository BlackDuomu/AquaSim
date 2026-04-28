from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_python(script: Path, args: Iterable[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(script), *list(args)]
    return subprocess.run(cmd, cwd=str(cwd), check=False, text=True)


def save_json(path: Path, obj: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(dict(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_cli_args(pairs: Mapping[str, Any]) -> List[str]:
    args: List[str] = []
    for k, v in pairs.items():
        key = f"--{k}"
        if isinstance(v, bool):
            if v:
                args.append(key)
            continue
        if v is None:
            continue
        args.extend([key, str(v)])
    return args


def parse_simple_yaml(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    stack: List[MutableMapping[str, Any]] = [out]
    level_stack = [0]

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, val = line.strip().split(":", 1)
        val = val.strip()

        while len(level_stack) > 1 and indent < level_stack[-1]:
            level_stack.pop()
            stack.pop()

        cur = stack[-1]
        if val == "":
            node: Dict[str, Any] = {}
            cur[key] = node
            stack.append(node)
            level_stack.append(indent + 2)
            continue

        low = val.lower()
        if low in {"true", "false"}:
            parsed: Any = low == "true"
        else:
            try:
                parsed = int(val)
            except ValueError:
                try:
                    parsed = float(val)
                except ValueError:
                    parsed = val.strip("'").strip('"')
        cur[key] = parsed
    return out


def dump_simple_yaml(path: Path, obj: Mapping[str, Any], indent: int = 0) -> None:
    lines: List[str] = []

    def _emit(prefix: str, value: Any, level: int) -> None:
        pad = " " * level
        if isinstance(value, Mapping):
            lines.append(f"{pad}{prefix}:")
            for k, v in value.items():
                _emit(str(k), v, level + 2)
            return
        if isinstance(value, bool):
            sval = "true" if value else "false"
        elif value is None:
            sval = "null"
        else:
            sval = str(value)
        lines.append(f"{pad}{prefix}: {sval}")

    for k, v in obj.items():
        _emit(str(k), v, indent)

    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_existing(preferred: Path, fallback: Path) -> Path:
    return preferred if preferred.exists() else fallback

