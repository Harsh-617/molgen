"""
visualizer.py

2D molecular structure rendering using RDKit.
Returns SVG strings and base64-encoded PNG images.
"""

from typing import Optional
import base64
import io

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def smiles_to_svg(
    smiles: str,
    width:  int = 300,
    height: int = 200,
) -> Optional[str]:
    """
    Render a SMILES string as an SVG string.

    Args:
        smiles: Valid SMILES string.
        width:  Image width in pixels.
        height: Image height in pixels.

    Returns:
        SVG string, or None if the molecule is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None


def smiles_to_image_b64(
    smiles: str,
    width:  int = 300,
    height: int = 200,
) -> Optional[str]:
    """
    Render a SMILES string as a base64-encoded PNG image.

    Args:
        smiles: Valid SMILES string.
        width:  Image width in pixels.
        height: Image height in pixels.

    Returns:
        Base64-encoded PNG string (no data URI prefix),
        or None if the molecule is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        img = Draw.MolToImage(mol, size=(width, height))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None