"""
Tools for MD scripts
"""

import pytraj as pt

from typing import Optional, Tuple

default_mask = ""


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
