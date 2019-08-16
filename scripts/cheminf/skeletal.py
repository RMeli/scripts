import argparse as ap

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

import os

from typing import List, Optional

def load_mol_from_file(fname: str):
    _, ext = os.path.splitext(fname)

    if ext == ".pdb":
        return Chem.MolFromPDBFile(fname)
    elif ext == ".mol2":
        return Chem.MolFromMol2File(fname)
    else:
        raise RuntimeError(f"Unsupported input format {ext}")

def skeletal(
    files: List[str],
    output: str,
    mols_per_row: int = 1,
    sub_img_width: int = 500,
    sub_img_height: int = 500,
    legend: bool = False,
):

    # Load molceules
    mols = [load_mol_from_file(file) for file in files]

    # Transform in 2D
    for mol in mols:
        AllChem.Compute2DCoords(mol)

    if legend:
        fnames = [os.path.basename(file) for file in files]
        names = [os.path.splitext(fname)[0] for fname in fnames]

    # Draw
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(sub_img_width, sub_img_height),
        legends=names if legend else None,
    )

    img.save(output)


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
    parser.add_argument("input", nargs="+", type=str)
    parser.add_argument("-o", "--output", default="mol.png", type=str)
    parser.add_argument("-mpr", "--molecules_per_row", default=4, type=int)
    parser.add_argument("-iw", "--width", default=500, type=int)
    parser.add_argument("-ih", "--height", default=500, type=int)
    parser.add_argument("-l", "--legend", action="store_true")

    # Parse arguments
    return parser.parse_args(args)


if __name__ == "__main__":

    args = parse()

    #args_dict = args_to_dict(args)

    skeletal(
        args.input,
        args.output,
        args.molecules_per_row,
        args.width,
        args.height,
        args.legend,
    )
