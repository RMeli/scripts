"""
Histogram plot
"""

import argparse as ap

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from typing import Optional


def plot(options: ap.Namespace) -> None:

    if len(options.input) != len(options.data):
        raise ValueError()

    plt.figure()

    # Check number of labels
    if options.labels is not None:
        if len(options.labels) != len(options.data):
            raise ValueError()

    # Set title and labels
    if options.title is not None:
        plt.title(options.title)
    if options.xlabel is not None:
        plt.xlabel(options.xlabel)

    # Load data
    data = np.loadtxt(options.input[0])
    for fname in options.input[1:]:
        d = np.loadtxt(fname)
        data = np.hstack((data, d))

    for i, idx in enumerate(options.data):

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
        )

    plt.xlim(left=options.left, right=options.right)

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

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    plot(args)
