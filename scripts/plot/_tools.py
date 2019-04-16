import numpy as np
import pandas as pd

import seaborn as sns

from collections import defaultdict
from typing import Iterable, Optional, List


def check_groups(groups: List[int]) -> int:

    # Check that group indices are sequential
    m = max(groups)
    for g in range(m):
        if g not in groups:
            raise ValueError("Group indices are not sequential.")

    return m + 1


def get_colormap(groups: Optional[List[int]] = None) -> np.ndarray:

    if groups is None:
        return sns.color_palette()

    # Color palette names
    names = ["Blues", "Reds", "Greens", "Purples", "Greys"]

    # Get group size (and check that group indices are consecutive)
    n = check_groups(groups)

    if n > len(names):
        raise ValueError("Too many groups for the available color palettes.")

    # Setup n color palettes, indexed by group
    # Get a MLP palette by name (as list of RGB values), reverse the color order
    # (with [::-1]) and make it iterable (so that next can be called later)
    palettes = {g: iter(sns.mpl_palette(names[g])[::-1]) for g in range(n)}

    colors = []
    for group in groups:
        colors.append(next(palettes[group]))

    return np.asarray(colors)