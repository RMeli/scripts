from scripts.md.molsplit import split_molecules

import scripts.md._tools as tools

import pytest
import os

path = os.path.abspath("tests/md/data/")


def test_split_protein_water():

    itraj = os.path.join(path, "alanine_dipeptide.nc")
    itop = os.path.join(path, "alanine_dipeptide.parm7")

    u = tools.load_traj_mda(itraj, itop)

    assert len(u.atoms) == 1912
    assert len(u.trajectory) == 25

    split = split_molecules(u)

    assert len(split["protein"].atoms) == 22
    assert len(split["water"].atoms) == 1912 - 22


def test_split_all():
    
    itraj = os.path.join(path, "1q72.mdcrd")
    itop = os.path.join(path, "1q72.parm7")
    
    u = tools.load_traj_mda(itraj, itop)

    split = split_molecules(u, keep_ions=True)

    print(split)

    assert len(split["protein"].atoms) == 6515

    assert len(split["water"].atoms) == 1500
    assert len(split["water"].residues) == 500

    assert len(split["LIG"].atoms) == 44

    assert len(split["Na+"].atoms) == 1


def test_split_noions():
    
    itraj = os.path.join(path, "1q72.mdcrd")
    itop = os.path.join(path, "1q72.parm7")
    
    u = tools.load_traj_mda(itraj, itop)

    split = split_molecules(u)

    print(split)

    assert len(split["protein"].atoms) == 6515

    assert len(split["water"].atoms) == 1500
    assert len(split["water"].residues) == 500

    assert len(split["LIG"].atoms) == 44

    with pytest.raises(KeyError):
        split["Na+"]