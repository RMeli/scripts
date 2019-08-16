"""
Plot Precision-Recall (PC) curve.

The PR curve is obtained by plotting the *precision*

.. math::
    \\text{P} = \\frac{\\text{T}_p}{\\text{T}_p + \\text{F}_p}

versus the *recall*

.. math::
    \\text{P} = \\frac{\\text{T}_p}{\\text{T}_p + \\text{F}_n}

where :math:`\\text{T}_p`, :math:`\\text{F}_p` and :math:`\\text{F}_pn` are the number
of true positives, false positive and false negatives, respectively.

The *average precision*

.. math::
    \\text{AP} = \\sum_{n} (R_n - R_{n-1})P_n,

where :math:`\\text{R}_n` is the recall tor the :math:`n`-th threshold and 
:math:`\\text{P}_n` is the precision tor the :math:`n`-th threshold, summarizes the PR 
curve as the weighted average of precisions achieved at each threshold. The increase in
recall is used as weight.

.. note::
    For K-fold cross validation, multiple PR curves can be plotted together.
"""

from scripts.plot._tools import get_colormap

import argparse as ap
import itertools
import numpy as np

from sklearn.metrics import precision_recall_curve, average_precision_score

from matplotlib import pyplot as plt

from typing import Tuple, List, Optional, Dict, Any, Union


def _pr_auc(fname: str, positive_label: Union[int, float]) -> Tuple[np.array, np.array, float]:
    """
    Generate PR curve and compute AUC

    Args:
        fname (str): Name of the data file

    Returns:
        Returns the false positive valuse (as ``np.array``), true positive values 
        (as ``np.array``) and the AUC (as ``float``).

    .. note::
        The data file :param:`fname` is a two-column file containing the class 
        ``y_true`` of the examples and their respective score ``y_score``.
    """

    # Load data from file
    y_true, y_score = np.loadtxt(fname, unpack=True)

    # Generate PR curve
    p, r, _ = precision_recall_curve(y_true, y_score, pos_label=positive_label)

    # Compute average precision score and average recall score
    aps = average_precision_score(y_true, y_score, pos_label=positive_label)

    return p, r, aps


def plot(
    fin: List[str],
    output: Optional[str] = None,
    groups: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
    positive_label: Union[int, float] = 1,
) -> None:
    """
    Plot PR curves.

    Args:
        fin (List[str]): List of input files
        output (str, optional): Output file for the plot

    Raises:
        ValueError: An error occurs when the number of labels is different from the 
        number of groups.

    .. note::
        If ``output`` is not specified, the plot is shown interactively.

    .. note::
        If ``groups==0`` each plot is considered as a different fold of the same group.
    """

    # Figure
    plt.figure()
    ax = plt.subplot(
        1,
        1,
        1,
        aspect="equal",
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title="Precision-Recall Curve" if title is None else title,
        xlabel="Precision",
        ylabel="Recall",
    )

    # Get color map
    cmap = get_colormap(groups)

    if labels is not None:
        if len(labels) != len(fin):
            raise ValueError(
                "The number of labels should be the same as the number of inputs."
            )

    for idx, f in enumerate(fin):
        prec, rec, aps = _pr_auc(f, positive_label)

        try:
            label = f"{labels[idx]} (AP = {aps:.2f})"
        except:
            label = f"AP = {aps:.2f}"

        # Plot PR
        ax.plot(prec, rec, label=label, color=cmap[idx])

    # Set legend
    ax.legend(loc="lower left")

    # Plot or save
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
        "fin": args.input,
        "output": args.output,
        "groups": args.groups,
        "labels": args.labels,
        "title": args.title,
        "positive_label": args.positive_label,
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
    parser = ap.ArgumentParser(description="Plot PR curve(s).")

    # Add arguments
    parser.add_argument("input", nargs="+", type=str)
    parser.add_argument("-o", "--output", default=None, type=str)
    parser.add_argument("-g", "--groups", nargs="*", default=None, type=int)
    parser.add_argument("-l", "--labels", nargs="*", default=None, type=str)
    parser.add_argument("-t", "--title", default=None, type=str)
    parser.add_argument("-pl", "--positive_label", default=1)

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    args_dict = args_to_dict(args)

    plot(**args_dict)
