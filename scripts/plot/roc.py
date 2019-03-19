"""
Plot Receiver Operating Characteristic (ROC) curve.

K-fold cross validation is supported.
"""

import argparse as ap
import numpy as np

from typing import List, Optional

from sklearn.metrics import roc_curve, roc_auc_score, auc

from matplotlib import pyplot as plt


def plot(fin: List[str], output: Optional[str] = None) -> None:
    """
    Plot ROC curves.

    Args:
        fin (List[str]): List of input files
        output (str, optional): Output file for the plot

    .. note::
        If ``output`` is not specified, the plot is shown interactively.
    """
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

    ax.plot([0, 1], [0, 1], "--", label="Random")

    if len(fin) == 1:
        y_true, y_score = np.loadtxt(fin[0], unpack=True)

        auc_score = roc_auc_score(y_true, y_score)

        fpr, tpr, ths = roc_curve(y_true, y_score)

        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")

    else:

        for idx, f in enumerate(fin):
            y_true, y_score = np.loadtxt(f, unpack=True)

            auc_score = roc_auc_score(y_true, y_score)

            fpr, tpr, ths = roc_curve(y_true, y_score)

            ax.plot(fpr, tpr, label=f"Fold {idx} (AUC = {auc_score:.2f})")

    ax.legend(loc="lower right")

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
    parser = ap.ArgumentParser(description="Plot ROC")

    # Add arguments
    parser.add_argument("-i", "--input", nargs="+")
    parser.add_argument("-o", "--output", default=None)

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    plot(args.input, args.output)
