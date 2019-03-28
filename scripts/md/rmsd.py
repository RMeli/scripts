"""
Trajectory RMSD
"""

import argparse as ap

import pytraj as pt
import numpy as np

from typing import Optional, Tuple

default_mask = ":LIG"


def _load_traj(
    itraj: str, itop: Optional[str] = None, mask: str = default_mask
) -> pt.Trajectory:
    """
    Load trajectory (and topology) from file.

    Args:
        itraj (str): Trajectory file name
        itop (str): Topology file name
        mask (str): Selection mask (in `pytraj` formart)

    Returns:
        Returns a `pt.Trajectory` as trajectory
    """

    traj = pt.load(itraj, itop, mask=mask)

    return traj


def _load_ref(
    iref: str, itop: Optional[str] = None, mask: str = default_mask
) -> pt.Frame:
    """
    Load reference structure (and topology) from file.

    Args:
        iref (str): Reference structure file name
        itop (str): Topology file name
        mask (str): Selection mask (in `pytraj` formart)

    Returns:
        Returns a `pt.Frame` as reference structure

    Raises:
        ValueError: An error occurs when the reference structure contains more than one
        frame.
    """

    ref = pt.load(iref, itop, mask=mask)

    if ref.n_frames != 1:
        raise ValueError("ERROR: Reference structure should contain only one frame.")

    return ref[0]


def _load(
    itraj: str,
    itop: Optional[str] = None,
    iref: Optional[str] = None,
    mask: str = default_mask,
) -> Tuple[pt.Trajectory, pt.Frame]:
    """
    Load reference structure (and topology) from file.

    Args:
        itraj (str): Trajectory file name
        itop (str): Topology file name
        iref (str): Reference structure file name
        mask (str): Selection mask (in `pytraj` formart)

    Returns:
        Returns a `pt.Trajectory` as trajectory and a `pt.Frame` as reference structure
    """

    traj = _load_traj(itraj, itop, mask)

    if iref is not None:
        ref = _load_ref(iref, itop, mask)
    else:
        ref = traj[0]

    return traj, ref


def compute_rmsd(
    itraj: str,
    itop: Optional[str] = None,
    iref: Optional[str] = None,
    mask: str = default_mask,
) -> np.ndarray:
    """
    Compute RMSD for a trajectory with respect to a reference structure.

    Args:
        itraj (str): Trajectory file name
        itop (str, optional): Topology file name
        iref (str, optional): Reference file name
        mask (str, optional): Selection mask (in `pytraj` format)

    Returns:
        Returns a `np.ndarray` containing the frame number, an the RMSD (in angstrom) 
        with respect to the reference structure `iref`.
    """

    traj, ref = _load(itraj, itop, iref, mask)

    # Compute RMSD (symmetrized)
    rmsd = pt.analysis.rmsd.symmrmsd(traj, fit=False)

    # TODO: Add time

    return np.stack((np.arange(0, traj.n_frames), rmsd), axis=1)


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
    parser.add_argument("-t", "--top", type=str, help="Topology file")
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

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    import numpy as np

    args = parse()

    # TODO: Check if files exist

    # Compute RMSD ([frame, time (ps), RMSD (A)])
    rmsd = compute_rmsd(args.traj, args.top, args.ref, args.mask)

    # Save RMSD to file
    np.savetxt(args.output, rmsd)

    if args.plot:
        raise NotImplementedError()
