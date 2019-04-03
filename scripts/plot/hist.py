"""
Histogram plot
"""

import argparse as ap

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from typing import Optional


def plot(options: ap.Namespace) -> None:
    """
    Plot univariate distribution

    Args:
        options (ap.Namespace): `ap.Namespace` with command-line options

    Raises:

    .. note:
        See the :func:`parser` for a description of all the possible command line
        arguments.

    .. note:
        The column index specified with `-d` (or `--data`) for every input file 
        correspond to the column index within that file, starting from 0.
    """

    if len(options.data) != len(options.input):
        raise ValueError("Inconsistent number of input files and data columns.")

    # Check number of groups
    g_max = len(options.input)  # One input per group
    groups = [0] * g_max  # All inputs in the same group by default
    if options.groups is not None:
        if len(options.groups) != len(options.input):
            raise ValueError("Inconsistent number of input files and groups.")

        # Check group index bounds
        for g in options.groups:
            if g < 0 or g >= g_max:
                raise ValueError(f"Group index {g} is out of bounds [{0},{g_max})")

        # Check that group indices are consecutive
        m = max(options.groups)
        for idx in range(m):
            if idx not in options.groups:
                raise ValueError(f"Group indices are not consecutive.")

        groups = options.groups

    n_plots = max(groups) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    # Make axes iterable for a single plot
    if n_plots == 1:
        axes = [axes]

    # Check number of labels
    if options.labels is not None:
        if len(options.labels) != len(options.data):
            raise ValueError()

    # Set title and labels
    if options.title is not None:
        fig.suptitle(options.title)
    if options.xlabel is not None:
        for ax in axes:
            ax.set_xlabel(options.xlabel)

    # Get colormap
    cm = sns.color_palette()

    for i, idx in enumerate(options.data):

        # Load data
        data = np.loadtxt(options.input[i])

        # Get label (if exists)
        hist_kws, kde_kws = None, None
        try:
            # Try to use options.labels as a list
            label = options.labels[i]

            if options.kde:
                kde_kws = {"label": label}
            else:
                hist_kws = {"label": label}

        except TypeError:  # If options.labels is not a list, a TypeError occurs
            # Do nothing (hist_kws=None, kde_kws=None)
            pass

        sns.distplot(
            data[:, idx],
            bins=options.bins,
            kde=options.kde,
            hist_kws=hist_kws,
            kde_kws=kde_kws,
            color=cm[i],
            ax=axes[groups[i]],
        )

    for ax in axes:
        ax.set_xlim(left=options.left, right=options.right)

    if options.labels is not None:
        plt.legend()

    if options.output is not None:
        plt.savefig(options.output)
    else:
        plt.show()


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

    plot(args)
