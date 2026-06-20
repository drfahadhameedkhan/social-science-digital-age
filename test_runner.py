#!/usr/bin/env python3
"""
Test runner to identify bugs in the repository
"""
import sys
import pytest

if __name__ == "__main__":
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--color=yes"
    ])
    sys.exit(exit_code)
