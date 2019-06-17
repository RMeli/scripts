"""
Tools for MD scripts
"""

import pytraj as pt

import os
from typing import Optional, Tuple

default_mask = ""


def load_traj(
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

    print(f'Loading trajectory {os.path.basename(itraj)} with mask "{mask}"...', end="")

    traj = pt.load(itraj, itop, mask=mask)

    print(" done")

    return traj


def load_ref(
    iref: str, itop: Optional[str] = None, mask: str = default_mask
) -> pt.Trajectory:
    """
    Load reference structure (and topology) from file.

    Args:
        iref (str): Reference structure file name
        itop (str): Topology file name
        mask (str): Selection mask (in `pytraj` formart)

    Returns:
        Returns a `pt.Trajectory` as reference structure

    Raises:
        ValueError: An error occurs when the reference structure contains more than one
        frame.
    """

    print(f'Loading reference {os.path.basename(iref)} with mask "{mask}"...', end="")

    ref = pt.load(iref, itop, mask=mask)

    if ref.n_frames != 1:
        raise ValueError(f"Reference structure contains {ref.n_frames} frames.")

    print(" done")

    return ref
