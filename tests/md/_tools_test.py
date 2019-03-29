import scripts.md._tools as tools

import pytraj as pt

import pytest
import os


path = os.path.abspath("tests/md/data/")


def test_load_traj():

    itraj = os.path.join(path, "alanine_dipeptide.nc")
    itop = os.path.join(path, "alanine_dipeptide.parm7")

    traj = tools.load_traj(itraj, itop)

    assert traj.n_frames == 25
    assert traj.n_atoms == 1912

    dipeptide = pt.select_atoms(":ALA,ACE,NME", traj.top)
    assert len(dipeptide) == 22

    water = pt.select_atoms(":WAT", traj.top)
    assert len(water) == traj.n_atoms - len(dipeptide)


def test_load_traj_mask():

    itraj = os.path.join(path, "alanine_dipeptide.nc")
    itop = os.path.join(path, "alanine_dipeptide.parm7")

    for mask in [":ALA,ACE,NME", "!:WAT"]:
        traj = tools.load_traj(itraj, itop, mask)

        assert traj.n_frames == 25
        assert traj.n_atoms == 22


def test_load_ref():

    itraj = os.path.join(path, "alanine_dipeptide.nc")
    iref = os.path.join(path, "alanine_dipeptide.ncrst")
    itop = os.path.join(path, "alanine_dipeptide.parm7")

    with pytest.raises(ValueError):
        ref = tools.load_ref(itraj, itop)

    ref = tools.load_ref(iref, itop)

    assert ref.n_frames == 1
    assert ref.n_atoms == 1912


def test_load_ref_mask():

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


def test_load():
    pass
