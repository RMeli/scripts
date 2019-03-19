"""
Plot Receiver Operating Characteristic (ROC) curve.

The ROC curve is obtained by plotting the *true positive rate* (TPR, also known as 
*sensitivity* or *recall*)

.. math::
    \\text{TPR} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}
    
versus the *false positive rate* (FPR, also known as *fall-out* or *probability of 
false alarm*)

.. math::
    \\text{FPR} = \\frac{\\text{FP}}{\\text{FP} + \\text{TN}}

where :math:`\\text{TP}`, :math:`\\text{TN}`, :math:`\\text{FP}`, and :math:`\\text{FN}`
are the number of true positives, true negatives, false positives, and false negatives,
respectively.

.. note::
    K-fold cross validation is supported.
"""

import argparse as ap
import numpy as np

from typing import Tuple, List, Optional

from sklearn.metrics import roc_curve, roc_auc_score, auc

from matplotlib import pyplot as plt


def _roc_auc(fname: str) -> Tuple[np.array, np.array, float]:
    """
    Generate ROC curve and compute AUC

    Args:
        fname (str): Name of the data file

    Returns:
        Returns the false positive valuse (as ``np.array``), true positive values 
        (as ``np.array``) and the AUC (as ``float``).

    .. note::
        The data file :param:`fname` is a two-column file containing the class ``y_true`` of 
        the examples and their respective score ``y_score``.
    """

    # Load data from file
    y_true, y_score = np.loadtxt(fname, unpack=True)

    # Compute AUC
    auc = roc_auc_score(y_true, y_score)

    # Generate ROC curve
    fpr, tpr, ths = roc_curve(y_true, y_score)

    return fpr, tpr, auc

def plot(fin: List[str], output: Optional[str] = None) -> None:
    """
    Plot ROC curves.

    Args:
        fin (List[str]): List of input files
        output (str, optional): Output file for the plot

    .. note::
        If ``output`` is not specified, the plot is shown interactively.
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
        title="Receiver Operating Characteristic",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )

    # Plot ROC for random classifier
    ax.plot([0, 1], [0, 1], "--", label="Random", color="grey", lw=0.5)

    if len(fin) == 1: # Only one ROC

        fpr, tpr, auc_score = _roc_auc(fin[0])

        # Plot ROC
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")

    else: # Multiple ROCs (for different folds)

        for idx, f in enumerate(fin): # Iterate over folds

            fpr, tpr, auc_score = _roc_auc(fin[0])

            # Plot ROC
            ax.plot(fpr, tpr, label=f"Fold {idx} (AUC = {auc_score:.2f})")

    # Set legend
    ax.legend(loc="lower right")

    # Plot or save
    if output is not None:
        plt.savefig(output)
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
    parser = ap.ArgumentParser(description="Plot ROC curve(s).")

    # Add arguments
    parser.add_argument("-i", "--input", nargs="+")
    parser.add_argument("-o", "--output", default=None)

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    plot(args.input, args.output)
