"""
Split different molecules
"""

import scripts.md._tools as tools 

import MDAnalysis as mda
import argparse as ap

import re

from typing import Dict, Optional, Any

def split_molecules(u: mda.Universe, keep_ions: bool=False) -> Dict[str, mda.AtomGroup]:

    split = {}

    # Select protein
    protein = u.select_atoms("protein")
    if len(protein.atoms) != 0:
        split["protein"] = protein

    # Select water molecules
    for water_name in ["WAT", "HOH"]:
        water = u.select_atoms(f"resname {water_name}")

        if len(water.atoms) != 0:
            break # If selection is not empty, stop
    if len(water.atoms) != 0:
        split["water"] = water

    # Other molecules
    other = u.select_atoms("all") - protein - water
    for res in other.residues: # Loop over all "other" residues
        name = res.resname
        if keep_ions or re.search("[A-Z]?[+-]", name) is None: # Remove or keep ions
            split[name] = res

    return split


def molsplit(itraj: str, itop: Optional[str] = None, keep_ions: bool = False) -> None:
    
    u = tools.load_traj_mda(itraj, itop)

    split = split_molecules(u, keep_ions)

    print(split)

    for name, atomgroup in split.items():
        fname = name + ".pdb"

        with mda.Writer(fname, multiframe=False) as W:
            W.write(atomgroup)


def args_to_dict(args: ap.Namespace) -> Dict[str, Any]:
    """
    Convert command line arguments to dictionary.

    Args:
        args (ap.Namespace): Command line arguments

    Returns:
        A dictionarty with kwargs and values
    """

    return {
        "itraj": args.input,
        "itop": args.topology,
        "keep_ions": args.keep_ions,
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
    parser = ap.ArgumentParser(description="Split molecules.")

    # Add arguments
    parser.add_argument("input", type=str, help="Input file")
    parser.add_argument("-t", "--topology", type=str, default=None, help="Topology file")
    parser.add_argument("-ki", "--keep_ions", action="store_true", help="Keep ions")

    # Parse arguments
    return parser.parse_args(args)

if __name__ == "__main__":
    
    args = parse()

    args_dict = args_to_dict(args)

    molsplit(**args_dict)