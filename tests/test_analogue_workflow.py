from pathlib import Path
from types import SimpleNamespace
import csv
import multiprocessing as mp
import time

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from matcha.analogue import AnalogueWorkflowConfig, run_analogue_workflow
from matcha.analogue import constrained_embed
from matcha.analogue.workflow import _gnina_rerank_poses, _select_final_ranked_poses
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


def test_constrained_embed_sets_rdkit_timeout():
    params = AllChem.ETKDGv3()

    constrained_embed._set_embed_timeout(params, 12)

    assert params.timeout == 12


def test_constrained_embed_process_watchdog_times_out(monkeypatch):
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("process watchdog test requires fork")

    def slow_embed(_mol, _num_conformers, _params):
        time.sleep(5)
        return []

    monkeypatch.setattr(constrained_embed, "_embed_multiple_confs_once", slow_embed)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = AllChem.ETKDGv3()

    with pytest.raises(TimeoutError):
        constrained_embed._embed_multiple_confs(mol, 1, params, timeout_seconds=0.2)


def test_final_pose_selection_keeps_best_and_fills_diverse():
    best = _mol3d("CCO", "best")
    duplicate = Chem.Mol(best)
    diverse = Chem.Mol(best)
    shifted = constrained_embed.mol_positions(diverse)
    shifted[:, 0] += 2.0
    constrained_embed.set_mol_positions(diverse, shifted)

    ranked = [
        (best, SimpleNamespace(rank_score=0.0)),
        (duplicate, SimpleNamespace(rank_score=1.0)),
        (diverse, SimpleNamespace(rank_score=2.0)),
    ]

    selected = _select_final_ranked_poses(ranked, n_final_poses=2, diversity_rmsd_cutoff=0.75)

    assert selected[0][0] is best
    assert selected[1][0] is diverse


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
            rbfe_pairwise_edges=False,
        ),
    )

    assert result.summary["ligands_with_seed_poses"] == 1
    assert result.summary["failed"] == 0
    assert result.summary["gnina_ranking"]["enabled"] is False
    assert result.summary["gnina_ranking"]["ranking_summary_csv"] is None
    assert result.summary["gnina_ranking"]["poses_scored"] == 0
    assert result.summary["rbfe_pairwise_edges"] is False
    assert result.summary["embed_oversample_factor"] == 4
    assert result.summary["embed_unconstrained_supplement"] is True
    assert result.summary["embed_seed_batches"] == 4
    assert result.summary["conformer_request_count"] >= 4
    assert result.summary["conformer_raw_count"] >= result.summary["conformer_after_dedup_count"] >= 1
    assert (tmp_path / "analogue_seed_transforms.npy").exists()
    assert (tmp_path / "fep_bundle_seed" / "aligned_series.sdf").exists()
    assert (tmp_path / "fep_bundle_seed" / "fep_manifest.json").exists()
    quality_report = (tmp_path / "fep_bundle_seed" / "quality_report.csv").read_text()
    assert "receptor_clash_count" in quality_report
    assert "receptor_contact_count" in quality_report

    transforms = np.load(tmp_path / "analogue_seed_transforms.npy", allow_pickle=True)[0]
    assert any(key.startswith("analogue_mol0_conf") for key in transforms)


def test_analogue_workflow_uses_gnina_scores_for_reranking(tmp_path: Path, monkeypatch):
    class FakeGninaScorer:
        def __init__(self, *, gnina_path, minimize, score_type, cnn_scoring, **_kwargs):
            assert gnina_path == "/bin/gnina"
            assert minimize is False
            assert score_type == "Affinity"
            assert cnn_scoring == "none"

        def score_poses(self, receptor_path, sdf_input_dir, sdf_output_dir, device=0):
            assert device == 0
            assert Path(receptor_path).exists()
            sdf_output_dir = Path(sdf_output_dir)
            sdf_output_dir.mkdir(parents=True, exist_ok=True)
            for sdf_path in Path(sdf_input_dir).glob("*.sdf"):
                writer = Chem.SDWriter(str(sdf_output_dir / sdf_path.name))
                writer.SetKekulize(False)
                for idx, mol in enumerate(Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)):
                    if mol is None:
                        continue
                    mol.SetProp("Affinity", f"{10 - idx:.3f}")
                    writer.write(mol)
                writer.close()

    import matcha.scoring.gnina_scorer as gnina_scorer

    monkeypatch.setattr(gnina_scorer, "GninaScorer", FakeGninaScorer)

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
            gnina_score_poses=True,
            gnina_scorer_path="/bin/gnina",
            gnina_minimize=False,
        ),
    )

    scored_sdf = tmp_path / "gnina_ranking" / "analogue" / "gnina_scored" / "analogue.sdf"
    scores = [
        float(mol.GetProp("Affinity"))
        for mol in Chem.SDMolSupplier(str(scored_sdf), removeHs=False, sanitize=False)
        if mol is not None
    ]

    selected = result.selected_molecules["analogue"]
    assert selected.HasProp("analogue_gnina_score")
    assert float(selected.GetProp("analogue_gnina_score")) == min(scores)
    assert scores[0] != min(scores)

    best_pose = next(
        Chem.SDMolSupplier(
            str(tmp_path / "fep_bundle_seed" / "complexes" / "analogue" / "best_pose.sdf"),
            removeHs=False,
            sanitize=False,
        )
    )
    assert best_pose.HasProp("analogue_gnina_score")
    assert float(best_pose.GetProp("analogue_gnina_score")) == min(scores)

    summary_rows = list(csv.DictReader(open(tmp_path / "gnina_ranking_summary.csv")))
    assert len(summary_rows) == len(scores)
    assert sum(row["selected"] == "True" for row in summary_rows) == 1
    selected_row = next(row for row in summary_rows if row["selected"] == "True")
    assert float(selected_row["gnina_score"]) == min(scores)
    assert result.summary["gnina_ranking"]["enabled"] is True
    assert result.summary["gnina_ranking"]["ligands_scored"] == 1
    assert result.summary["gnina_ranking"]["poses_scored"] == len(scores)
    assert result.summary["gnina_ranking"]["poses_missing_score"] == 0
    assert result.summary["gnina_ranking"]["selected_changed_by_gnina"] == 1


def test_gnina_reranking_preserves_fep_ready_before_affinity(tmp_path: Path, monkeypatch):
    class FakeGninaScorer:
        def __init__(self, **_kwargs):
            pass

        def score_poses(self, _receptor_path, sdf_input_dir, sdf_output_dir, device=0):
            sdf_output_dir = Path(sdf_output_dir)
            sdf_output_dir.mkdir(parents=True, exist_ok=True)
            for sdf_path in Path(sdf_input_dir).glob("*.sdf"):
                writer = Chem.SDWriter(str(sdf_output_dir / sdf_path.name))
                writer.SetKekulize(False)
                for idx, mol in enumerate(Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)):
                    if mol is None:
                        continue
                    mol.SetProp("Affinity", "-100.0" if idx == 1 else "100.0")
                    writer.write(mol)
                writer.close()

    def fake_qc(pose_idx):
        fep_ready = pose_idx == 0
        return SimpleNamespace(
            ligand_id="analogue",
            pose_index=pose_idx,
            core_rmsd=0.1 if fep_ready else 4.0,
            receptor_clash_count=0,
            receptor_contact_count=0,
            rank_score=0.0,
            status="FEP_READY" if fep_ready else "NEEDS_REVIEW",
            fep_ready=fep_ready,
            warnings=[],
        )

    import matcha.scoring.gnina_scorer as gnina_scorer

    monkeypatch.setattr(gnina_scorer, "GninaScorer", FakeGninaScorer)

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  C   ALA A   1       8.000   8.000   8.000  1.00  0.00           C\n"
        "END\n"
    )
    ranked = [(_mol3d("Cc1ccccc1", "ready"), fake_qc(0)), (_mol3d("CCc1ccccc1", "review"), fake_qc(1))]

    reranked = _gnina_rerank_poses(
        ligand_id="analogue",
        ranked=ranked,
        template=_mol3d("Cc1ccccc1", "template"),
        mapping=None,
        receptor_path=receptor,
        receptor_positions=None,
        output_dir=tmp_path / "gnina",
        cfg=AnalogueWorkflowConfig(
            gnina_score_poses=True,
            gnina_scorer_path="/bin/gnina",
            gnina_minimize=False,
        ),
    )

    assert reranked[0][1].status == "FEP_READY"
    assert reranked[0][0].GetProp("analogue_gnina_score") == "100.000000"
    assert reranked[1][1].status == "NEEDS_REVIEW"
    assert reranked[1][0].GetProp("analogue_gnina_score") == "-100.000000"


def test_gnina_reranking_uses_higher_cnn_scores(tmp_path: Path, monkeypatch):
    class FakeGninaScorer:
        def __init__(self, *, score_type, cnn_scoring, **_kwargs):
            assert score_type == "CNNscore"
            assert cnn_scoring == "rescore"

        def score_poses(self, _receptor_path, sdf_input_dir, sdf_output_dir, device=0):
            sdf_output_dir = Path(sdf_output_dir)
            sdf_output_dir.mkdir(parents=True, exist_ok=True)
            for sdf_path in Path(sdf_input_dir).glob("*.sdf"):
                writer = Chem.SDWriter(str(sdf_output_dir / sdf_path.name))
                writer.SetKekulize(False)
                for idx, mol in enumerate(Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)):
                    if mol is None:
                        continue
                    mol.SetProp("CNNscore", "0.1" if idx == 0 else "0.9")
                    writer.write(mol)
                writer.close()

    def fake_qc(pose_idx):
        return SimpleNamespace(
            ligand_id="analogue",
            pose_index=pose_idx,
            core_rmsd=0.1,
            receptor_clash_count=0,
            receptor_contact_count=0,
            rank_score=0.0,
            status="FEP_READY",
            fep_ready=True,
            warnings=[],
        )

    import matcha.scoring.gnina_scorer as gnina_scorer

    monkeypatch.setattr(gnina_scorer, "GninaScorer", FakeGninaScorer)

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  C   ALA A   1       8.000   8.000   8.000  1.00  0.00           C\n"
        "END\n"
    )
    ranked = [(_mol3d("Cc1ccccc1", "low"), fake_qc(0)), (_mol3d("CCc1ccccc1", "high"), fake_qc(1))]

    reranked = _gnina_rerank_poses(
        ligand_id="analogue",
        ranked=ranked,
        template=_mol3d("Cc1ccccc1", "template"),
        mapping=None,
        receptor_path=receptor,
        receptor_positions=None,
        output_dir=tmp_path / "gnina",
        cfg=AnalogueWorkflowConfig(
            gnina_score_poses=True,
            gnina_scorer_path="/bin/gnina",
            gnina_minimize=False,
            gnina_score_type="CNNscore",
            gnina_cnn_scoring="rescore",
        ),
    )

    assert reranked[0][1].pose_index == 1
    assert reranked[0][0].GetProp("analogue_gnina_score") == "0.900000"
