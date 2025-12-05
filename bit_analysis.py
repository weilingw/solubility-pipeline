import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np
import re
import math
from io import BytesIO
# Draw and save image
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DCairo
from PIL import Image, ImageDraw, ImageFont
# from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdmolops


def sanitize_filename(name):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def create_bit_summary_visualization(bit_index, matched_solutes, df, solute_bitinfos, output_dir, sub_img_size=(600, 600)):
    """
    Create a summary visualization for a specific bit using row indices (not solute names).
    """
    from rdkit.Chem import Draw

    sub_mols = []
    legends = []
    highlight_atom_lists = []
    highlight_bond_lists = []

    seen_envs = set()

    for idx in matched_solutes:
        solute_smiles = df.loc[idx, 'solute_smiles']
        solute_mol = Chem.MolFromSmiles(solute_smiles)
        bit_info = solute_bitinfos[idx]

        if bit_index in bit_info:
            for atom_id, radius in bit_info[bit_index]:
                env = Chem.FindAtomEnvironmentOfRadiusN(solute_mol, radius, atom_id)
                if not env:
                    continue

                atoms = []
                for bond_idx in env:
                    bond = solute_mol.GetBondWithIdx(bond_idx)
                    atoms.extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                highlight_atoms = list(set([atom_id] + atoms))
                highlight_bonds = env

                env_key = (tuple(sorted(highlight_atoms)), tuple(sorted(highlight_bonds)))
                if env_key in seen_envs:
                    continue
                seen_envs.add(env_key)

                sub_mols.append(solute_mol)
                legends.append(f"Row {idx} (Bit {bit_index})")
                highlight_atom_lists.append(highlight_atoms)
                highlight_bond_lists.append(highlight_bonds)

    if sub_mols:
        img = Draw.MolsToGridImage(
            sub_mols,
            molsPerRow=5,
            subImgSize=sub_img_size,
            legends=legends,
            highlightAtomLists=highlight_atom_lists,
            highlightBondLists=highlight_bond_lists
        )

        os.makedirs(output_dir, exist_ok=True)
        png_path  = os.path.join(output_dir, f"global_bit_{bit_index}_summary.png")
        tiff_path = os.path.join(output_dir, f"global_bit_{bit_index}_summary.tiff")

        img.save(png_path)
        img.save(tiff_path, format='TIFF', dpi=(600, 600), compression='tiff_lzw')


def summarize_bit_frequency(df, top_bits, output_dir, role=None):
    """
    Count how many rows activate each bit in top_bits.
    top_bits: iterable of (role_str, bit_id) tuples, e.g. ('solute', 993)
    role: if provided ('solute' or 'solvent'), only summarize that role.
    """
    rows = []
    for role_, b in top_bits:
        if (role is None) or (role_ == role):
            col = f"{role_}_FP_{b}"
            if col in df.columns:
                rows.append({
                    "bit": int(b),
                    "role": role_,
                    "count": int(df[col].sum())
                })

    if not rows:
        print(f"No rows to summarize in summarize_bit_frequency (role={role}).")
        return

    out_name = f"{role or 'all'}_bit_usage_summary.csv"
    (pd.DataFrame(rows)
       .sort_values(["role", "count"], ascending=[True, False])
       .to_csv(os.path.join(output_dir, out_name), index=False))


def create_solute_summary_visualization(solute_name, matched_bits, df, solute_bitinfos, output_dir):
    idx = df[df['solute_name'] == solute_name].index[0]
    mol = Chem.MolFromSmiles(df.loc[idx, 'solute_smiles'])
    bit_info = solute_bitinfos[idx]

    sub_mols = []
    legends = []
    highlight_atom_lists = []
    highlight_bond_lists = []
    seen_envs = set()

    for bit in matched_bits:
        if bit in bit_info:
            for atom_id, radius in bit_info[bit]:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_id)
                if not env:
                    continue
                atoms = []
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms.extend([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                highlight_atoms = list(set([atom_id] + atoms))
                highlight_bonds = env
                env_key = (tuple(sorted(highlight_atoms)), tuple(sorted(highlight_bonds)))
                if env_key in seen_envs:
                    continue
                seen_envs.add(env_key)
                sub_mols.append(mol)
                legends.append(f"Bit {bit} (atom {atom_id}, radius {radius})")
                highlight_atom_lists.append(highlight_atoms)
                highlight_bond_lists.append(highlight_bonds)

    if sub_mols:
        img = Draw.MolsToGridImage(
            sub_mols, molsPerRow=5, subImgSize=(600, 600), legends=legends,
            highlightAtomLists=highlight_atom_lists, highlightBondLists=highlight_bond_lists
        )
        os.makedirs(output_dir, exist_ok=True)
        png_path  = os.path.join(output_dir, f"{sanitize_filename(solute_name)}_summary.png")
        tiff_path = os.path.join(output_dir, f"{sanitize_filename(solute_name)}_summary.tiff")
        img.save(png_path)
        img.save(tiff_path, format='TIFF', dpi=(600, 600), compression='tiff_lzw')


def visualize_top_bits_with_bitinfo(
    top_bits, df, smiles_col, bit_mapping, output_dir, role, bitinfo_data, sub_img_size=(600, 600)
):
    """
    For each top bit, generate a substructure summary across all solutes/solvents that activate it.
    bitinfo_data is now indexed by row number, not SMILES.
    """
    print(f"\nVisualizing top bits for {role} using index-based bitInfo...")

    print(" Sample rows in DataFrame:", df[[smiles_col]].head())
    print(" Sample keys in bitinfo_data (row indices):", list(bitinfo_data.keys())[:5])

    for role_, local_bit in top_bits:
        col = f"{role_}_FP_{local_bit}"
        if col not in df.columns:
            print(f" Bit column {col} missing from DataFrame — skipping.")
            continue

        activated = df[col].sum()
        print(f" Bit {local_bit} ({role_}) activated in {int(activated)} molecule(s).")

        matched_rows = []
        bitinfos_for_matches = {}

        seen_smiles = set()

        for idx, row in df.iterrows():
            if idx not in bitinfo_data:
                continue

            mol_bitinfo = bitinfo_data[idx]
            if local_bit not in mol_bitinfo:
                continue

            smi = row[smiles_col]
            if smi in seen_smiles:
                continue  # only one structure per unique SMILES

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            seen_smiles.add(smi)

            highlight_atoms = set()
            for center_idx, radius in mol_bitinfo[local_bit]:
                env = rdmolops.FindAtomEnvironmentOfRadiusN(mol, radius, center_idx)
                atoms_in_env = {center_idx}  # to include the centre atom
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms_in_env.update([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                highlight_atoms.update(atoms_in_env)
            highlight_atoms = list(highlight_atoms)


            # Get bonds between highlighted atoms
            highlight_bonds = []
            atom_set = set(highlight_atoms)
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                if a1 in atom_set and a2 in atom_set:
                    highlight_bonds.append(bond.GetIdx())

            # Create subfolder for each bit
            bit_dir = os.path.join(output_dir, f"bit_{local_bit}")
            os.makedirs(bit_dir, exist_ok=True)

            # Define image output path
            img_path = os.path.join(bit_dir, f"{idx}_bit_{local_bit}_{sanitize_filename(smi)}.png")


            # Assign a consistent highlight color to all atoms and bonds
            highlight_color = (1.0, 0.5, 0.5)
            highlight_atom_colors = {idx: highlight_color for idx in highlight_atoms}
            highlight_bond_colors = {idx: highlight_color for idx in highlight_bonds}

            # Ensure 2D coordinates are computed
            rdDepictor.Compute2DCoords(mol)
            drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)
            rdMolDraw2D.PrepareAndDrawMolecule(
                drawer,
                mol,
                highlightAtoms=highlight_atoms,
                highlightBonds=highlight_bonds,
                highlightAtomColors=highlight_atom_colors,
                highlightBondColors=highlight_bond_colors
            )
            drawer.FinishDrawing()

            png_bytes = drawer.GetDrawingText()
            with open(img_path, "wb") as f:
                f.write(png_bytes)  
                
            im = Image.open(BytesIO(png_bytes))
            im.save(img_path.replace(".png", ".tiff"), format="TIFF", dpi=(600, 600), compression = 'tiff_lzw')
            im.close()
            # Record for summary
            matched_rows.append(idx)
            bitinfos_for_matches[idx] = mol_bitinfo[local_bit]

        if matched_rows:
            print(f"Bit {local_bit} matched in {len(matched_rows)} molecules — visualizing.")
            create_bit_summary_visualization(
                bit_index=local_bit,
                matched_solutes=matched_rows,
                df=df,
                solute_bitinfos=bitinfos_for_matches,
                output_dir=output_dir,
                sub_img_size=sub_img_size
            )
        else:
            print(f"Bit {local_bit} was not active in any molecule — skipping.")








