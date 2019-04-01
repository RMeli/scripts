"""
Reimage trajectory with periodic boundary conditions (PBC)
"""

import argparse as ap

import pytraj as pt

from scripts.md._tools import load_traj

from typing import Optional


def _save_traj(ofname: str, otraj: pt.Trajectory, overwrite: bool = True) -> None:

    pt.save(ofname, otraj, overwrite=overwrite)


def reimage(itraj: str, ofname: str, itop: Optional[str] = None) -> None:

    traj = load_traj(itraj, itop)

    traj = pt.autoimage(traj)

    traj = pt.image(traj, "origin center protein")

    _save_traj(ofname, traj)


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
    parser.add_argument(
        "-x", "--traj", type=str, required=True, help="Input trajectory file"
    )
    parser.add_argument(
        "-o", "--out", type=str, required=True, help="Output Trajectory file"
    )
    parser.add_argument("-t", "--top", type=str, default=None, help="Topology file")

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    reimage(args.traj, args.out, args.top)
