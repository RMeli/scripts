import scripts.md._tools as tools

import pytraj as pt

import pytest
import os


path = os.path.abspath("tests/md/data/")


def test_load_traj():
    """
    Load analine dipeptide trajectory without mask
    """

    itraj = os.path.join(path, "alanine_dipeptide.nc")
    itop = os.path.join(path, "alanine_dipeptide.parm7")

    traj = tools.load_traj(itraj, itop)

    assert traj.n_frames == 25
    assert traj.n_atoms == 1912

    dipeptide = pt.select_atoms(":ALA,ACE,NME", traj.top)
    assert len(dipeptide) == 22

    water = pt.select_atoms(":WAT", traj.top)
    assert len(water) == traj.n_atoms - len(dipeptide)


def test_load_pdb():
    """
    Load benzene PDB file without mask
    """

    itraj = os.path.join(path, "benzene.pdb")

    traj = tools.load_traj(itraj)

    assert traj.n_frames == 1
    assert traj.n_atoms == 12

    C = pt.select_atoms("@C", traj.top)
    assert len(C) == 6

    H = pt.select_atoms("@H", traj.top)
    assert len(H) == 6


def test_load_traj_mask():
    """
    Load analine dipeptide trajectory with masks
    """

    itraj = os.path.join(path, "alanine_dipeptide.nc")
    itop = os.path.join(path, "alanine_dipeptide.parm7")

    for mask in [":ALA,ACE,NME", "!:WAT"]:
        traj = tools.load_traj(itraj, itop, mask)

        assert traj.n_frames == 25
        assert traj.n_atoms == 22


def test_load_pdb_mask():
    """
    Load benzene PDB file with mask
    """

    itraj = os.path.join(path, "benzene.pdb")

    traj = tools.load_traj(itraj, mask="@C")

    assert traj.n_frames == 1
    assert traj.n_atoms == 6

    H = pt.select_atoms("@H", traj.top)
    assert len(H) == 0


def test_load_ref():
    """
    Load reference structure (single frame)
    """

    itraj = os.path.join(path, "alanine_dipeptide.nc")
    iref = os.path.join(path, "alanine_dipeptide.ncrst")
    itop = os.path.join(path, "alanine_dipeptide.parm7")

    with pytest.raises(ValueError):
        ref = tools.load_ref(itraj, itop)

    ref = tools.load_ref(iref, itop)

    assert ref.n_frames == 1
    assert ref.n_atoms == 1912


def test_load_ref_mask():
    """
    Load reference structure (single frame) with masks
    """

    itraj = os.path.join(path, "alanine_dipeptide.nc")
    iref = os.path.join(path, "alanine_dipeptide.ncrst")
    itop = os.path.join(path, "alanine_dipeptide.parm7")

    for mask in [":ALA,ACE,NME", "!:WAT"]:
        traj = tools.load_traj(itraj, itop, mask)

        with pytest.raises(ValueError):
            ref = tools.load_ref(itraj, itop, mask)

        ref = tools.load_ref(iref, itop, mask)

        assert ref.n_frames == 1
        assert ref.n_atoms == 22


def test_load_traj_mda():

    itraj = os.path.join(path, "alanine_dipeptide.nc")
    itop = os.path.join(path, "alanine_dipeptide.parm7")

    u = tools.load_traj_mda(itraj, itop)

    assert len(u.atoms) == 1912
    assert len(u.trajectory) == 25

    peptide = u.select_atoms("protein")

    assert len(peptide.atoms) == 22
    assert len(peptide.residues) == 3

    water = u.select_atoms("resname WAT")

    assert len(water.atoms) == len(u.atoms) - len(peptide.atoms)