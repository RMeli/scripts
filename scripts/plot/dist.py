"""
Distribution plot (with kernel density estimation).
"""

from scripts.plot._tools import get_colormap

import argparse as ap

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from typing import Optional, List, Dict, Any

def plot(
    input: List[str],
    data: List[int],
    output: Optional[str] = None,
    bins: int = 50,
    kde: bool = False,
    left: Optional[float] = None,
    right: Optional[float] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    labels: Optional[List[str]] = None,
    groups: Optional[List[int]] = None
) -> None:
    """
    Plot univariate distribution (with kernel density estimation)

    Args:
        input (List[str]): Input file(s) name(s)
        data (List[int]): Column index (indices) for data
        output (str, optional): Output file name
        bins (int, optional): Number of bins
        kde (bool, optional): Perform kernel density estimation
        left (float, optional): Lower x-axis limit
        right (float, optional): Higher x-axis limit
        title (str, optional): Plot title
        xlabel (str, optional): x-axis label
        ylabel (str, optional): y-axis label
        labels (List[str], optional): List of labels for legend
        groups (List[int], optional): Group indices for different subplots

    Raises:

    .. note:
        See the :func:`parser` for a description of all the possible command line
        arguments.

    .. note:
        The column index specified with `-d` (or `--data`) for every input file 
        correspond to the column index within that file, starting from 0.
    """

    if len(data) != len(input):
        raise ValueError("Inconsistent number of input files and data columns.")

    # Check number of groups
    g_max = len(input)  # One input per group
    groups_default = [0] * g_max  # All inputs in the same group by default
    if groups is not None:
        if len(groups) != len(input):
            raise ValueError("Inconsistent number of input files and groups.")

        # Check group index bounds
        for g in groups:
            if g < 0 or g >= g_max:
                raise ValueError(f"Group index {g} is out of bounds [{0},{g_max})")

        # Check that group indices are consecutive
        m = max(groups)
        for idx in range(m):
            if idx not in groups:
                raise ValueError(f"Group indices are not consecutive.")

    groups = groups_default if groups is None else groups

    n_plots = max(groups) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # Make axes iterable for a single plot
    if n_plots == 1:
        axes = [axes]

    # Check number of labels
    if labels is not None:
        if len(labels) != len(data):
            raise ValueError()

    # Set title and labels
    if title is not None:
        fig.suptitle(title)
    if xlabel is not None:
        for ax in axes:
            ax.set_xlabel(xlabel)

    # Get colormap
    cm = get_colormap()

    for i, idx in enumerate(data):

        # Load data
        d = np.loadtxt(input[i])

        # Get label (if exists)
        hist_kws, kde_kws = None, None
        try:
            # Try to use options.labels as a list
            label = labels[i]

            if kde:
                kde_kws = {"label": label}
            else:
                hist_kws = {"label": label}

        except TypeError:  # If labels is not a list, a TypeError occurs
            # Do nothing (hist_kws=None, kde_kws=None)
            pass

        sns.distplot(
            d[:, idx],
            bins=bins,
            kde=kde,
            hist_kws=hist_kws,
            kde_kws=kde_kws,
            color=cm[i],
            ax=axes[groups[i]],
        )

    for ax in axes:
        ax.set_xlim(left=left, right=right)

    if labels is not None:
        plt.legend()

    if output is not None:
        plt.savefig(output)
    else:
        plt.show()


def args_to_dict(args: ap.Namespace) -> Dict[str, Any]:
    """
    Convert command line arguments to dictionary.

    Args:
        args (ap.Namespace): Command line arguments

    Returns:
        A dictionarty with kwargs and values
    """

    return {
        "input": args.input,
        "data": args.data,
        "output": args.output,
        "bins": args.bins,
        "kde": args.kde,
        "left": args.left,
        "right": args.right,
        "title": args.title,
        "xlabel": args.xlabel,
        "ylabel": args.ylabel,
        "labels": args.labels,
        "groups": args.groups,
    }

def parse(args: Optional[str] = None) -> ap.Namespace:
    """
    Parse command-line arguments.

    Args:
        args (str, optional): String to parse
    
    Returns:
        An `ap.Namespace` containing the parsed options

    .. note::
        If ``args is None`` the string to parse is red from ``sys.argv``
    """

    # Parser
    parser = ap.ArgumentParser(description="Histogram plot.")

    # Add arguments
    parser.add_argument("-i", "--input", nargs="+", type=str, help="Input file")
    parser.add_argument(
        "-d", "--data", nargs="+", type=int, help="Column index (indices) of the data"
    )
    parser.add_argument(
        "-b", "--bins", default=None, type=int, help="Number of histogram beans"
    )
    parser.add_argument(
        "--kde", default=False, action="store_true", help="Kernel Density Estimate"
    )
    parser.add_argument("-ll", "--left", default=None, type=float, help="Left limit")
    parser.add_argument("-rl", "--right", default=None, type=float, help="Right limit")
    parser.add_argument("-o", "--output", default=None, type=str, help="Output file")
    parser.add_argument("-t", "--title", default=None, type=str, help="Plot title")
    parser.add_argument("-lx", "--xlabel", default=None, type=str, help="x-axis label")
    parser.add_argument("-ly", "--ylabel", default=None, type=str, help="y-axis label")
    parser.add_argument(
        "-l", "--labels", nargs="*", default=None, type=str, help="Labels"
    )
    parser.add_argument(
        "-g", "--groups", nargs="*", default=None, type=int, help="Group indices"
    )

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    args_dict = args_to_dict(args)

    plot(**args_dict)
