from scripts.md.rmsd import compute_rmsd

import numpy as np

import pytest
import os

path = os.path.abspath("tests/md/data/")


def test_rmsd_benzene_superimposed():

    benzene = os.path.join(path, "benzene.pdb")

    rmsd = compute_rmsd(benzene, lig_mask="")

    assert rmsd.shape == (1, 2)  # Only one frame
    assert rmsd[0, 0] == pytest.approx(0.0)  # Frame 0
    assert rmsd[0, 1] == pytest.approx(0.0)  # RMSD: 0


def test_rmsd_benzene_shifted():

    benzene = os.path.join(path, "benzene.pdb")
    benzene_s = os.path.join(path, "benzene_shifted.pdb")

    rmsd = compute_rmsd(benzene, iref=benzene_s, lig_mask="")

    assert rmsd.shape == (1, 2)  # Only one frame
    assert rmsd[0, 0] == pytest.approx(0.0)  # Frame 0
    assert rmsd[0, 1] == pytest.approx(1.0)  # RMSD: 0


def test_rmsd_noWAT_noH():

    traj = os.path.join(path, "alanine_dipeptide.nc")
    ref = os.path.join(path, "alanine_dipeptide.ncrst")
    top = os.path.join(path, "alanine_dipeptide.parm7")

    rmsd = compute_rmsd(traj, itop=top, iref=ref, lig_mask="!:WAT&!@H=")

    _, rmsd_cpptraj = np.loadtxt(os.path.join(path, "alanine_dipeptide.rmsd"), unpack=True)

    for idx, (_, r) in enumerate(rmsd):
        assert r == pytest.approx(rmsd_cpptraj[idx], abs=1e-3)
