#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark import main as benchmark_main

    return benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
