"""
Box-and-Whisker Plot
"""

import scripts.plot._tools as tools

import argparse as ap
import os
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

from typing import Dict, Any, Optional


def load_dataframe(ifname: str) -> pd.DataFrame:

    # Get file extension
    _, ext = os.path.splitext(ifname)

    # Clean extension
    ext = ext.lower().strip()

    if ext == ".csv":
        df = pd.read_csv(ifname)
    else:
        raise IOError(f"Unsupported file extension {ext}")

    return df


def plot(
    df: pd.DataFrame,
    output: Optional[str] = None,
    x_name: Optional[str] = None,
    y_name: Optional[str] = None,
    hue_name: Optional[str] = None,
    swarm: bool = False,
    notch: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    legend_name: Optional[str] = None,
    hide_fliers: bool = False,
) -> None:

    # TODO: Compute and show the average as well?

    # if remove_outliers:
    #    df = tools.remove_outliers(df)

    if hide_fliers:
        # Do not show outliers
        showfliers = False
    else:
        # Do not show outliers if the swarm is plotted
        showfliers = not swarm

    sns.boxplot(
        data=df, x=x_name, y=y_name, hue=hue_name, notch=notch, showfliers=showfliers
    )

    if swarm:
        ax = sns.swarmplot(
            data=df,
            x=x_name,
            y=y_name,
            hue=hue_name,
            dodge=True,
            edgecolor="gray",  # Color lines around each point
            linewidth=1,
        )

        # Fix legend for duplicates
        # The legend contains both the hues from sns.boxplot than the ones from
        # sns.swarmplot
        h, l = ax.get_legend_handles_labels()
        n = len(h) // 2
        plt.legend(h[:n], l[:n], title=legend_name)
    else:
        # A legend is produced only when hue is given
        if hue_name is not None:
            plt.legend(title=legend_name)

    # Title and labels
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Plot or save
    if output is not None:
        plt.savefig(output)
    else:
        plt.show()


def args_to_dict(args: ap.Namespace) -> Dict[str, Any]:
    """

    .. note
        This functions load a `pd.DataFrame` from file, since the input is a file name
        but `plot` requires a `pd.DataFrame`.
    """

    df = load_dataframe(args.input)

    return {
        "df": df,
        "output": args.output,
        "x_name": args.xname,
        "y_name": args.yname,
        "hue_name": args.group,
        "swarm": args.swarm,
        "notch": args.notch,
        "title": args.title,
        "xlabel": args.xlabel,
        "ylabel": args.ylabel,
        "legend_name": args.legend,
        "hide_fliers": args.hide_fliers,
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
    parser.add_argument("-i", "--input", type=str, required=True, help="Input file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")
    parser.add_argument("-x", "--xname", type=str, default=None, help="x name")
    parser.add_argument("-y", "--yname", type=str, default=None, help="y name")
    parser.add_argument("-g", "--group", type=str, default=None, help="Hue name")
    parser.add_argument(
        "-s", "--swarm", default=False, action="store_true", help="Swarmplot"
    )
    parser.add_argument(
        "-n", "--notch", default=False, action="store_true", help="Notch"
    )
    parser.add_argument("-t", "--title", type=str, default=None, help="Plot title")
    parser.add_argument("-lx", "--xlabel", type=str, default=None, help="x label")
    parser.add_argument("-ly", "--ylabel", type=str, default=None, help="y label")
    parser.add_argument("-ln", "--legend", type=str, default=None, help="Legend name")
    parser.add_argument(
        "--hide-fliers", default=False, action="store_true", help="Do not show outliers"
    )

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    args_dict = args_to_dict(args)

    plot(**args_dict)
