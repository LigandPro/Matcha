import copy
import numpy as np

import networkx as nx
from networkx.algorithms import isomorphism as iso
from rdkit import Chem
from Bio.PDB import PDBParser, MMCIFParser, PDBIO, Select

from flowdock.utils.preprocessing import read_molecule


def restore_atom_order(ref_mol, pred_mol):
    ref_graph = mol_to_nx(ref_mol)
    pred_graph = mol_to_nx(pred_mol)
    def node_match(attr1, attr2):
        return attr1["atomic_num"] == attr2["atomic_num"]
    gm = iso.GraphMatcher(ref_graph, pred_graph, node_match=node_match)
    if gm.is_isomorphic():
        mapping = gm.mapping
        ordered_mapping = [mapping[i] for i in range(ref_mol.GetNumAtoms())]
        new_pred_mol = Chem.rdmolops.RenumberAtoms(pred_mol, ordered_mapping)
        ref_mol_copy = copy.deepcopy(ref_mol)
        ref_mol_copy.GetConformer().SetPositions(new_pred_mol.GetConformer().GetPositions())
        return ref_mol_copy
    return pred_mol


def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), atomic_num=atom.GetAtomicNum(), atom=atom)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        G.add_edge(i, j)
    return G
    

def filter_protein_chains_by_ligand_distance(reference_protein_pdb_path, reference_ligand_path, 
                                             output_protein_pdb_path, distance_cutoff=10, return_all=False):
    # Read the reference ligand
    ligand_mol = read_molecule(reference_ligand_path)
    if ligand_mol is None:
        raise ValueError(f"Could not read ligand from {reference_ligand_path}")
    
    # Get ligand coordinates
    conf = ligand_mol.GetConformer()
    ligand_coords = conf.GetPositions()
    
    # Read the reference protein structure
    if reference_protein_pdb_path.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        parser = MMCIFParser(QUIET=True)
    protein_structure = parser.get_structure('protein', reference_protein_pdb_path)
    
    # Calculate distances between ligand and protein chains
    kept_chains = []
    chain_distances = {}
    chain_atom_counts = {}
    
    for chain in protein_structure.get_chains():
        chain_id = chain.get_id()
        chain_coords = []
        
        # Collect all atom coordinates from this chain
        for residue in chain:
            for atom in residue:
                if atom.get_name() != 'H':  # Skip hydrogens for distance calculation
                    chain_coords.append(atom.get_coord())
        
        if len(chain_coords) == 0:
            print(f"Chain {chain_id}: No atoms found")
            continue
            
        chain_coords = np.array(chain_coords)
        
        # Calculate minimum distance between ligand and this chain
        distances = np.linalg.norm(ligand_coords[:, None] - chain_coords[None, :], axis=-1)
        min_distance = distances.min()
        chain_distances[chain_id] = min_distance
        atoms_within_cutoff = np.sum(distances.min(axis=0) < distance_cutoff)
        chain_atom_counts[chain_id] = atoms_within_cutoff

    if return_all:
        kept_chains = [chain_id for chain_id, count in chain_atom_counts.items() if count > 0]
    else:
        primary_chain = max(chain_atom_counts.items(), key=lambda x: x[1])[0]
        kept_chains = [primary_chain]

    # Create a filtered structure with only the kept chains
    class ChainSelector(Select):
        def __init__(self, kept_chains):
            self.kept_chains = kept_chains
            
        def accept_chain(self, chain):
            return chain.get_id() in self.kept_chains
    
    for chain_id in kept_chains:
        if return_all:
            chain_output_protein_pdb_path = output_protein_pdb_path.replace('.pdb', f'_{chain_id}.pdb')
        else:
            chain_output_protein_pdb_path = output_protein_pdb_path
        # Save the filtered protein structure
        io = PDBIO()
        io.set_structure(protein_structure)
        io.save(chain_output_protein_pdb_path, ChainSelector([chain_id]))
    
    if len(kept_chains) == 0:
        print("WARNING: No chains were kept! Consider increasing the distance cutoff.")
        print("Chain distances to ligand:")
        for chain_id, distance in sorted(chain_distances.items()):
            print(f"  {chain_id}: {distance:.2f} Å")
    
    return kept_chains


def align_to_binding_site(
    predicted_protein: str,
    predicted_ligand: str,
    reference_protein: str,
    reference_ligand: str,
    aligned_ligand_path: str,
    aligned_protein_path: str,
    cutoff: float = 10.0,
):
    """Align the predicted protein-ligand complex to the reference complex
    using the reference protein's heavy atom ligand binding site residues.

    :param predicted_protein: File path to the predicted protein (PDB).
    :param predicted_ligand: File path to the optional predicted ligand
        (SDF).
    :param reference_protein: File path to the reference protein (PDB).
    :param reference_ligand: File path to the optional reference ligand
        (SDF).
    :param dataset: Dataset name (e.g., "dockgen", "casp15",
        "posebusters_benchmark", or "astex_diverse").
    :param aligned_filename_suffix: Suffix to append to the aligned
        files (default "_aligned").
    :param cutoff: Distance cutoff in Å to define the binding site
        (default 10.0).
    :param save_protein: Whether to save the aligned protein structure
        (default True).
    :param save_ligand: Whether to save the aligned ligand structure
        (default True).
    """
    from pymol import cmd

    # Initialize PyMOL
    cmd.delete("all")
    cmd.reinitialize()

    # Load structures
    cmd.load(reference_protein, "ref_protein")
    cmd.load(predicted_protein, "pred_protein")

    cmd.load(reference_ligand, "ref_ligand")
    cmd.load(predicted_ligand, "pred_ligand")

    # Group predicted protein and ligand(s) together for alignment
    cmd.create(
        "pred_complex",
        ("pred_protein or pred_ligand" if predicted_ligand is not None else "pred_protein"),
    )

    # Select heavy atoms in the reference protein
    cmd.select("ref_protein_heavy", "ref_protein and not elem H")

    # Select heavy atoms in the reference ligand(s)
    cmd.select("ref_ligand_heavy", "ref_ligand and not elem H")

    # Define the reference binding site(s) based on the reference ligand(s)
    cmd.select("binding_site", f"(backbone) and ref_protein_heavy within {cutoff} of ref_ligand_heavy")

    # Align the predicted protein to the reference binding site(s)
    alignment_result = cmd.align("pred_complex", "binding_site") #, cycles=0)

    # Apply the transformation to the individual objects
    cmd.matrix_copy("pred_complex", "pred_protein")
    cmd.matrix_copy("pred_complex", "pred_ligand")

    # Save the aligned ligand
    cmd.save(aligned_ligand_path, "pred_ligand")
    cmd.save(aligned_protein_path, "pred_protein")

    # Clean up
    cmd.delete("all")

    return alignment_result[0]


def align_to_binding_site_by_pocket(
    predicted_protein: str,
    predicted_ligand: str,
    reference_protein: str,
    reference_ligand: str,
    aligned_ligand_path: str,
    aligned_protein_path: str,
    cutoff: float = 10.0,
):
    """Align the predicted protein-ligand complex to the reference complex
    using the reference protein's heavy atom ligand binding site residues.

    :param predicted_protein: File path to the predicted protein (PDB).
    :param predicted_ligand: File path to the optional predicted ligand
        (SDF).
    :param reference_protein: File path to the reference protein (PDB).
    :param reference_ligand: File path to the optional reference ligand
        (SDF).
    :param dataset: Dataset name (e.g., "dockgen", "casp15",
        "posebusters_benchmark", or "astex_diverse").
    :param aligned_filename_suffix: Suffix to append to the aligned
        files (default "_aligned").
    :param cutoff: Distance cutoff in Å to define the binding site
        (default 10.0).
    :param save_protein: Whether to save the aligned protein structure
        (default True).
    :param save_ligand: Whether to save the aligned ligand structure
        (default True).
    """
    from pymol import cmd

    # Initialize PyMOL
    cmd.delete("all")
    cmd.reinitialize()

    # Load structures
    cmd.load(reference_protein, "ref_protein")
    cmd.load(predicted_protein, "pred_protein")
    cmd.load(reference_ligand, "ref_ligand")
    cmd.load(predicted_ligand, "pred_ligand")

    # Select heavy atoms in the reference protein
    cmd.select("ref_protein_heavy", "ref_protein and not elem H")
    cmd.select("pred_protein_heavy", "pred_protein and not elem H")

    # Select heavy atoms in the reference ligand(s)
    cmd.select("ref_ligand_heavy", "ref_ligand and not elem H")
    cmd.select("pred_ligand_heavy", "pred_ligand and not elem H")

    # Define the reference binding site(s) based on the reference ligand(s)
    cmd.select("ref_binding_site", f"(backbone) and ref_protein_heavy within {cutoff} of ref_ligand_heavy")

    # Define the predicted binding site(s) based on the reference ligand(s)
    cmd.select("pred_binding_site", f"(backbone) and pred_protein_heavy within {cutoff} of pred_ligand_heavy")

    # Align the predicted protein to the reference binding site(s)
    alignment_result = cmd.align("pred_binding_site", "ref_binding_site", cycles=0)

    # Get the transformation matrix from the aligned protein
    transformation_matrix = cmd.get_object_matrix("pred_protein")

    # Apply the same transformation to the predicted ligand
    cmd.transform_selection("pred_ligand", transformation_matrix)

    # Save the aligned ligand
    cmd.save(aligned_ligand_path, "pred_ligand")
    cmd.save(aligned_protein_path, "pred_protein")

    # Clean up
    cmd.delete("all")

    return alignment_result[0]
