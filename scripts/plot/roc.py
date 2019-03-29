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

import argparse as ap
import itertools
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score, auc

from matplotlib import pyplot as plt

from typing import Tuple, List, Optional


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


def _get_colormap(n: int, groups: int, c_min: float = 0.3, c_max: float = 0.8):
    """
    Get colormap for different groups

    Each group (which contains the same number of folds) is colored with a different
    color.

    Args:
        n (int): Total number of plots
        groups (int): Number of groups
        c_min (float, optional): Minimum color range
        c_max (float, optional): Maximum color range

    Returns:
        Returns a colormap, i.e. a Numpy array containing RGB values for the colors,
        ordered by group.

    Raises:
        ValueError: An error occurs when the number of groups is higher than 4.
        ValueError: An error occurs when the number of plots is not a multiple of the
            number of groups.
    """

    if groups == 0:
        return plt.cm.tab10(np.linspace(0, 1, n))

    # Check number of groups
    if groups > 4:
        raise ValueError("A maximum of 4 groups is supported.")

    # Check that all groups have the same number of elements
    if n % groups != 0:
        raise ValueError(
            "The number of plots should be a multiple of the number of groups."
        )

    plots_per_group = n // groups

    cmap_names = ["Blues", "Reds", "Greens", "Purples"]

    c_range = np.linspace(c_min, c_max, plots_per_group)

    cmap = plt.cm.get_cmap(cmap_names[0])(c_range)
    for idx in range(1, groups):
        colors = plt.cm.get_cmap(cmap_names[idx])(c_range)
        cmap = np.concatenate((cmap, colors))

    return cmap


def plot(fin: List[str], output: Optional[str] = None, groups=0, labels=None) -> None:
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

    n = len(fin)

    cmap = _get_colormap(n, groups)

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

    if n == 1:  # Only one ROC

        fpr, tpr, auc_score = _roc_auc(fin[0])

        # Plot ROC
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color=cmap[0])

    else:  # Multiple ROCs

        # Compute number of folds
        if groups == 0:
            n_folds = n
        else:
            n_folds = n // groups

        # Cyclic iterator over folds
        fold = itertools.cycle(range(n_folds))

        if labels is not None:
            if len(labels) != groups:
                raise ValueError(
                    "The number of labels should be the same as the number of groups."
                )

            labels = [l for label in labels for l in itertools.repeat(label, n_folds)]

        for idx, f in enumerate(fin):  # Iterate over folds

            fpr, tpr, auc_score = _roc_auc(f)

            label = f"Fold {next(fold)} (AUC = {auc_score:.2f})"

            if labels is not None:
                label = f"{labels[idx]} " + label

            # Plot ROC
            ax.plot(fpr, tpr, label=label, color=cmap[idx])

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
    parser.add_argument("-i", "--input", nargs="+", type=str, required=True)
    parser.add_argument("-o", "--output", default=None, type=str)
    parser.add_argument("-g", "--groups", default=0, type=int)
    parser.add_argument("-l", "--labels", nargs="*", default=None, type=str)

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    plot(args.input, args.output, args.groups, args.labels)
