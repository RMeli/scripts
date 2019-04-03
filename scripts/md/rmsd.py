"""
Trajectory RMSD
"""

import argparse as ap
import sys

import pytraj as pt
import numpy as np

from scripts.md._tools import load_traj, load_ref

from typing import Optional


default_mask = ":LIG"


def compute_rmsd(
    itraj: str,
    itop: Optional[str] = None,
    iref: Optional[str] = None,
    lig_mask: str = default_mask,
    reimage: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute RMSD for a trajectory with respect to a reference structure.

    Args:
        itraj (str): Trajectory file name
        itop (str, optional): Topology file name
        iref (str, optional): Reference file name
        lig_mask (str, optional): Selection mask (in `pytraj` format) for the ligand
        reimage (bool): Re-image coordinates according to PBC

    Returns:
        Returns a `np.ndarray` containing the frame number, an the RMSD (in angstrom) 
        with respect to the reference structure `iref`.
    """

    if verbose:
        print("Loading trajectory...", file=sys.stdout, end="")

    lig_traj = load_traj(itraj, itop, mask=lig_mask)

    if verbose:
        print("done", file=sys.stdout)

    if iref is not None:
        if verbose:
            print("Loading reference...", file=sys.stdout, end="")

        lig_ref = load_ref(iref, itop, mask=lig_mask)

        if verbose:
            print("done", file=sys.stdout)
    else:
        lig_ref = 0

    # Autoimage (for PBC)
    if reimage:
        if verbose:
            print("Reimaging...", file=sys.stdout, end="")

        lig_traj = pt.autoimage(lig_traj)

        if iref is not None:
            lig_ref = pt.autoimage(lig_ref)

        if verbose:
            print("done", file=sys.stdout)

    # TODO: Align trajectory with reference structure
    # (needs to load the whole trajectory)

    # Compute RMSD (symmetrized)
    if verbose:
        print("Computing RMSD...", file=sys.stdout, end="")

    rmsd = pt.analysis.rmsd.symmrmsd(
        lig_traj, mask=lig_mask, ref=lig_ref, ref_mask=lig_mask, fit=False
    )

    if verbose:
        print("done", file=sys.stdout)

    # TODO: Add time

    return np.stack((np.arange(0, lig_traj.n_frames), rmsd), axis=1)


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
    parser.add_argument("-x", "--traj", type=str, required=True, help="Trajectory file")
    parser.add_argument("-t", "--top", type=str, default=None, help="Topology file")
    parser.add_argument("-r", "--ref", type=str, default=None, help="Reference file")
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        default=default_mask,
        help="Atom or residue mask (pytraj format)",
    )
    parser.add_argument(
        "-o", "--output", type=str, required=True, help="RMSD output file"
    )
    parser.add_argument("--plot", action="store_true", help="Plot RMSD vs time")
    parser.add_argument(
        "--reimage", action="store_true", help="Re-image trajectory within PBC box"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose execution"
    )

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":
    import os

    args = parse()

    if not os.path.isfile(args.traj):
        raise FileNotFoundError(args.traj)
    if args.top is not None and not os.path.isfile(args.top):
        raise FileNotFoundError(args.top)
    if args.ref is not None and not os.path.isfile(args.ref):
        raise FileNotFoundError(args.ref)

    # Compute RMSD ([frame, RMSD (A)])
    rmsd = compute_rmsd(
        args.traj, args.top, args.ref, args.mask, args.reimage, args.verbose
    )

    # Save RMSD to file
    np.savetxt(args.output, rmsd)

    if args.plot:
        raise NotImplementedError()
