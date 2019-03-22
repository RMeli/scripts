import argparse as ap
import numpy as np

from matplotlib import pyplot as plt

from typing import Optional


def plot(options: ap.Namespace) -> None:
    """
    Plot from command line arguments

    Args:
        options (ap.Namespace): `ap.Namespace` with command-line options

    Raises:
        ValueError: The number of labels does not correspond to the number of plots.

    .. note:
        See the :func:`parser` for a description of all the possible command line
        arguments.
    """

    plt.figure()

    # Check number of labels
    if options.labels is not None and len(options.ycoord) != len(options.labels):
        raise ValueError("Incorrect number of labels.")

    # Set title and labels
    if options.title is not None:
        plt.title(options.title)
    if options.xlabel is not None:
        plt.xlabel(options.xlabel)
    if options.ylabel is not None:
        plt.ylabel(options.ylabel)

    # Load data
    data = np.loadtxt(options.input[0])
    for fname in options.input[1:]:
        d = np.loadtxt(fname)
        data = np.hstack((data, d))

    # Number of rows
    n = data.shape[0]

    if options.xcoord is None:
        x = np.arange(0, n)
    else:
        x = data[:, options.xcoord]
    x = x * options.xmult

    for i, idx in enumerate(options.ycoord):
        y = data[:, idx] * options.ymult

        if options.labels is not None:
            plt.plot(x, y, label=options.labels[i])
        else:
            plt.plot(x, y)

    # Set limits
    if options.xlim is not None:
        plt.xlim(options.xlim)
    if options.ylim is not None:
        plt.ylim(options.ylim)

    # Activate legend
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
    parser = ap.ArgumentParser(description="Standard XY plot.")

    # Add arguments
    parser.add_argument("-i", "--input", nargs="+", type=str, help="Input file(s)")
    parser.add_argument("-o", "--output", default=None, type=str, help="Output file")
    parser.add_argument(
        "-x", "--xcoord", default=None, type=int, help="Column index of x-coordinate"
    )
    parser.add_argument(
        "-y",
        "--ycoord",
        nargs="+",
        type=int,
        help="Column index (indices) of y-coordinate(s)",
    )
    parser.add_argument(
        "-xl",
        "--xlim",
        nargs=2,
        default=None,
        type=float,
        help="Limits for x-coordinate",
    )
    parser.add_argument(
        "-yl",
        "--ylim",
        nargs=2,
        default=None,
        type=float,
        help="Limits for y-coordinate(s)",
    )
    parser.add_argument(
        "-l", "--labels", nargs="*", default=None, type=str, help="Labels"
    )
    parser.add_argument("-t", "--title", default=None, type=str, help="Plot title")
    parser.add_argument("-lx", "--xlabel", default=None, type=str, help="x-axis label")
    parser.add_argument("-ly", "--ylabel", default=None, type=str, help="y-axis label")
    parser.add_argument("-xm", "--xmult", default=1.0, type=float, help="x-value(s) multiplier")
    parser.add_argument("-ym", "--ymult", default=1.0, type=float, help="y-value(s) multiplier")

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    plot(args)
