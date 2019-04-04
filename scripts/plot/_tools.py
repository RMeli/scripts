import seaborn as sns

from collections import defaultdict
from typing import Iterable, Optional, List


def get_colormap(groups: Optional[List[int]] = None):

    if groups is None:
        return sns.color_palette()

    # TODO: Check no more than 4 element per group are present

    palette = sns.color_palette("tab20c")

    colors = []
    n = defaultdict(int)
    for group in groups:
        colors.append(palette[4 * group + n[group]])

        n[group] += 1

    return colors
