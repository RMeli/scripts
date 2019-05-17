"""
Plot Receiver Operating Characteristic (ROC) curve.

The ROC curve, a caracteristic of a binary classifier, is obtained by plotting the 
*true positive rate* (TPR, also known as *sensitivity* or *recall*)

.. math::
    \\text{TPR} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}
    
versus the *false positive rate* (FPR, also known as *fall-out* or *probability of 
false alarm*)

.. math::
    \\text{FPR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}}

where :math:`\\text{TP}`, :math:`\\text{TN}`, :math:`\\text{FP}`, and :math:`\\text{FN}`
are the number of true positives, true negatives, false positives, and false negatives,
respectively.

Here, the ROC curve is plotted from the true binary labels (:math:`[0,1]` or 
:math:`[-1,1]`) and the target scores (as probability estimates or confidence values).

.. note::
    For K-fold cross validation, multiple ROC curves can be plotted together.
"""

from scripts.plot._tools import get_colormap

import argparse as ap
import itertools
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score, auc

from matplotlib import pyplot as plt

from typing import Tuple, List, Optional, Dict, Any


def _roc_auc(fname: str) -> Tuple[np.array, np.array, float]:
    """
    Generate ROC curve and compute AUC

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

    # Compute AUC
    auc = roc_auc_score(y_true, y_score)

    # Generate ROC curve
    fpr, tpr, ths = roc_curve(y_true, y_score)

    return fpr, tpr, auc


def plot(
    fin: List[str],
    output: Optional[str] = None,
    groups: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot ROC curves.

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
        title="Receiver Operating Characteristic" if title is None else title,
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )

    # Get color map
    cmap = get_colormap(groups)

    # Plot ROC for random classifier
    ax.plot([0, 1], [0, 1], "--", label="Random", color="grey", lw=0.5)

    if labels is not None:
        if len(labels) != len(fin):
            raise ValueError(
                "The number of labels should be the same as the number of inputs."
            )

    for idx, f in enumerate(fin):
        fpr, tpr, auc_score = _roc_auc(f)

        try:
            label = f"{labels[idx]} (AUC = {auc_score:.2f})"
        except:
            label = f"AUC = {auc_score:.2f}"

        # Plot ROC
        ax.plot(fpr, tpr, label=label, color=cmap[idx])

    # Set legend
    ax.legend(loc="lower right")

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
    parser = ap.ArgumentParser(description="Plot ROC curve(s).")

    # Add arguments
    parser.add_argument("-i", "--input", nargs="+", type=str, required=True)
    parser.add_argument("-o", "--output", default=None, type=str)
    parser.add_argument("-g", "--groups", nargs="*", default=None, type=int)
    parser.add_argument("-l", "--labels", nargs="*", default=None, type=str)
    parser.add_argument("-t", "--title", default=None, type=str)

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    args_dict = args_to_dict(args)

    plot(**args_dict)
