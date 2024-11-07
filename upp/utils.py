from __future__ import annotations

import numpy as np

def path_append(path, suffix):
    return path.parent / f"{path.stem}_{suffix}{path.suffix}"
