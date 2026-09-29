"""Claim a result file without overwriting an earlier run's file."""
import os


def claim(path):
    """Create path exclusively and return it; if it exists, use <root>-2<ext>, <root>-3<ext>, ... instead."""
    root, ext = os.path.splitext(path)
    candidate, n = path, 1
    while True:
        try:
            with open(candidate, "x"):
                return candidate
        except FileExistsError:
            n += 1
            candidate = f"{root}-{n}{ext}"
