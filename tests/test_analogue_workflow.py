from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from matcha.analogue import AnalogueWorkflowConfig, run_analogue_workflow
from matcha.analogue.mcs import find_robust_mcs
from matcha.analogue.standardize import standardize_mol


def _mol3d(smiles: str, name: str):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=7)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    mol.UpdatePropertyCache(False)
    Chem.GetSymmSSSR(mol)
    mol.SetProp("_Name", name)
    return mol


def test_robust_mcs_maps_congeneric_core():
    template = standardize_mol(_mol3d("Cc1ccccc1", "template")).mol
    analogue = standardize_mol(_mol3d("CCc1ccccc1", "analogue")).mol

    mapping = find_robust_mcs(template, analogue, min_atoms=5, min_fraction=0.3, timeout=2)

    assert mapping.ok
    assert mapping.num_atoms >= 6
    assert mapping.fraction_ligand >= 0.75


def test_analogue_workflow_writes_fep_bundle(tmp_path: Path):
    template = _mol3d("Cc1ccccc1", "template")
    analogue = _mol3d("CCc1ccccc1", "analogue")
    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  C   ALA A   1       8.000   8.000   8.000  1.00  0.00           C\n"
        "END\n"
    )

    result = run_analogue_workflow(
        template_mol=template,
        ligands=[("analogue", analogue)],
        output_dir=tmp_path,
        receptor_path=receptor,
        config=AnalogueWorkflowConfig(
            n_seed_poses=4,
            n_final_poses=2,
            min_mcs_atoms=5,
            min_mcs_fraction=0.3,
            core_rmsd_cutoff=1.0,
        ),
    )

    assert result.summary["ligands_with_seed_poses"] == 1
    assert result.summary["failed"] == 0
    assert (tmp_path / "analogue_seed_transforms.npy").exists()
    assert (tmp_path / "fep_bundle_seed" / "aligned_series.sdf").exists()
    assert (tmp_path / "fep_bundle_seed" / "fep_manifest.json").exists()
    quality_report = (tmp_path / "fep_bundle_seed" / "quality_report.csv").read_text()
    assert "receptor_clash_count" in quality_report
    assert "receptor_contact_count" in quality_report

    transforms = np.load(tmp_path / "analogue_seed_transforms.npy", allow_pickle=True)[0]
    assert any(key.startswith("analogue_mol0_conf") for key in transforms)
