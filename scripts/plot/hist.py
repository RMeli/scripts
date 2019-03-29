"""
Histogram plot
"""

import argparse as ap

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt

from typing import Optional

def plot(options: ap.Namespace) -> None:

    plt.figure()

    # Set title and labels
    if options.title is not None:
        plt.title(options.title)
    if options.xlabel is not None:
        plt.xlabel(options.xlabel)

    # Load data
    data = np.loadtxt(options.input)

    sns.distplot(data[:, options.data], bins=options.bins, kde=options.kde)

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
    parser.add_argument("-i", "--input",required=True, type=str, help="Input file")
    parser.add_argument(
        "-d", "--data", required=True, type=int, help="Column index of the data"
    )
    parser.add_argument("-b", "--bins", default=None, type=int, help="Number of histogram beans")
    parser.add_argument(
        "--kde", default=False, action="store_true", help="Kernel Density Estimate"
    )
    parser.add_argument("-o", "--output", default=None, type=str, help="Output file")
    parser.add_argument("-t", "--title", default=None, type=str, help="Plot title")
    parser.add_argument("-lx", "--xlabel", default=None, type=str, help="x-axis label")
    parser.add_argument("-ly", "--ylabel", default=None, type=str, help="y-axis label")

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    plot(args)
