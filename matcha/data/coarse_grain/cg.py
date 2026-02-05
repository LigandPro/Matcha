from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Sequence, cast

import numpy as np
import pandas as pd

from matcha.utils.log import logger

from .pdb import PDBAtom, PDBRecord, PDBRepresentation, ResidueKey


class CGError(Exception):
    pass


@dataclass(frozen=True, kw_only=True)
class CGBead(PDBAtom):
    """
    A coarse-grained bead represented as a specialization of a PDB atom record.


    >>> line = "ATOM      1  N   ASP A   1      11.104  13.207   2.100  1.00 20.00"
    >>> atom = PDBAtom.from_pdb_line(line)
    >>> bead = CGBead(
    ...     record=PDBRecord.ATOM,
    ...     name="BB",
    ...     residue_name="ASP",
    ...     chain_id="A",
    ...     residue_id=1,
    ...     insertion_code=None,
    ...     x=0.0,
    ...     y=0.0,
    ...     z=0.0,
    ...     bead_type="BB",
    ...     n_atoms_used=1,
    ...     n_atoms_expected=1,
    ...     mass=atom.mass,
    ... )
    >>> bead.n_atoms_used
    1
    >>> bead.element is None
    True
    >>> bead.mass
    14.007...
    """

    bead_type: str
    n_atoms_used: int = field(default=0, kw_only=True)
    n_atoms_expected: int = field(default=0, kw_only=True)
    mass: float = field(default=0.0, kw_only=True)


def mass_center(atoms: Collection[PDBAtom]) -> tuple[float, float, float]:
    """
    Return the mass-weighted center of mass of atoms.

    Atoms with unknown/invalid mass are ignored. Returns `None` when there are no atoms
    with a valid mass.

    >>> line = "ATOM      1  N   ASP A   1      11.104  13.207   2.100  1.00 20.00           N"
    >>> atom = PDBAtom.from_pdb_line(line)
    >>> mass_center([atom])
    (11.104, 13.207, 2.1)
    """
    coords = np.asarray([(a.x, a.y, a.z) for a in atoms], dtype=float)
    masses = np.asarray([a.mass for a in atoms], dtype=float)
    total = float(masses.sum())
    center = (coords * masses[:, None]).sum(axis=0) / total
    return float(center[0]), float(center[1]), float(center[2])


@dataclass
class CGBuilder:
    """Stream a .pdb line-by-line and build CG beads using mass-weighted COM.

    For each residue, for each bead mapping (bead_type -> atom names), the bead is
    emitted if at least one mapped atom is present in that residue.
    """

    mappings: dict[str, dict[str, Sequence[str] | set[str]]]
    accept_alternate_location_codes: Sequence[str] = field(default=("", "A"), kw_only=True)
    first_model_only: bool = field(default=True, kw_only=True)
    accept_alternative_locations: tuple[str, ...] = field(init=False)
    ignore_missing_hydrogens: bool = True
    ignore_residue_names: list[str] = field(default_factory=list, kw_only=True)

    def __post_init__(self) -> None:
        self.accept_alternative_locations = tuple(
            a.strip().upper() for a in self.accept_alternate_location_codes
        )

    def _pick_altloc(self, candidates: Sequence[PDBAtom]) -> PDBAtom:
        if len(candidates) == 1:
            return candidates[0]
        for alt in self.accept_alternative_locations:
            for cand in candidates:
                if (cand.alternative_location_code or "").strip().upper() == alt:
                    return cand
        return candidates[0]

    def _select_atoms_for_bead(
        self, atoms_by_name: dict[str, list[PDBAtom]], atom_names: Sequence[str] | set[str]
    ) -> tuple[list[PDBAtom], list[str]]:
        selected: list[PDBAtom] = []
        missing: list[str] = []
        for atom_name in (str(a).strip() for a in atom_names):
            candidates = atoms_by_name.get(atom_name, [])
            if not candidates:
                if (
                    self.ignore_missing_hydrogens
                    and PDBAtom._element_from_atom_name(atom_name) == "H"
                ):
                    continue
                missing.append(atom_name)
                continue
            selected.append(self._pick_altloc(candidates))
        return selected, missing

    def build_beads(
        self,
        pdb_representation: PDBRepresentation,
        *,
        pdb_source: Path,
    ) -> list[CGBead]:
        beads: list[CGBead] = []

        # NOTE: Iterate over residues
        for residue_key, atom_group in PDBRepresentation.group_atoms_by_residue(pdb_representation):
            # Sanity check
            if not atom_group:
                raise CGError("Empty atom group encountered")

            # Silently skip residue if in ignore list
            if residue_key.residue_name in self.ignore_residue_names:
                continue

            # Check if we have a mapping for this residue
            if not (mapping := self.mappings.get(residue_key.residue_name)):
                logger.debug(
                    f"No beads created as no mapping found for residue {residue_key} "
                    f"in PDB {pdb_source}:{atom_group[0].line_no}"
                )
                continue

            # Define a lookup from atom name (key) to list of atoms with that name (value)
            name_to_atoms: defaultdict[str, list[PDBAtom]] = defaultdict(list)
            for atom in atom_group:
                name_to_atoms[atom.name].append(atom)

            # NOTE: Iterate over bead types
            for bead_type, atom_names_required in mapping.items():
                # Select atoms for the bead type
                atoms_found, atoms_missing = self._select_atoms_for_bead(
                    name_to_atoms, atom_names_required
                )

                if not atoms_found:
                    logger.debug(
                        f"No bead created: no atoms found for bead {bead_type!r} "
                        f"in residue {residue_key} in PDB {pdb_source}:{atom_group[0].line_no}"
                    )
                    continue

                # Compute center of mass and mass
                x, y, z = mass_center(atoms_found)
                mass = float(sum((a.mass or 0.0) for a in atoms_found))

                # Add bead instance to CG representation
                bead = CGBead(
                    record=PDBRecord.ATOM,
                    name=bead_type,
                    residue_name=residue_key.residue_name,
                    chain_id=residue_key.chain_id,
                    residue_id=residue_key.residue_id,
                    insertion_code=residue_key.insertion_code,
                    x=x,
                    y=y,
                    z=z,
                    bead_type=str(bead_type),
                    n_atoms_used=len(atoms_found),
                    n_atoms_expected=len(atom_names_required),
                    mass=mass,
                )
                beads.append(bead)

                if atoms_missing:

                    def is_hydrogen_atom_name(atom_name: str) -> bool:
                        return PDBAtom._element_from_atom_name(atom_name).upper() == "H"

                    expected_names = [str(name).strip() for name in atom_names_required]
                    missing_names = [str(name).strip() for name in atoms_missing]

                    if self.ignore_missing_hydrogens:
                        coverage_mode = "heavy"
                        used_count = sum(
                            1 for atom in atoms_found if (atom.element or "").strip().upper() != "H"
                        )
                        expected_count = sum(
                            1 for name in expected_names if not is_hydrogen_atom_name(name)
                        )
                        missing_reported = [
                            name for name in missing_names if not is_hydrogen_atom_name(name)
                        ]
                    else:
                        coverage_mode = "including hydrogen"
                        used_count = len(atoms_found)
                        expected_count = len(expected_names)
                        missing_reported = missing_names

                    logger.debug(
                        f"Bead created from partial atom-set: for {bead_type!r} found {used_count}/{expected_count} ({coverage_mode}) atoms "
                        f"(missing: {missing_reported}) "
                        f"in residue {residue_key} in PDB {pdb_source}:{atom_group[0].line_no}"
                    )

        return beads


@dataclass
class CGRepresentation(PDBRepresentation):
    @classmethod
    def from_full_atom_pdb(
        cls,
        pdb_representation: PDBRepresentation,
        pdb_source: Path,
        mappings: dict[str, dict[str, Sequence[str] | set[str]]],
        *,
        accept_alternate_location_codes: Sequence[str] = ("", "A"),
        first_model_only: bool = True,
        check_sequences: bool = True,
        check_geometry: bool = True,
    ) -> "CGRepresentation":
        """
        Build a CG representation from a *full-atom* PDB using `CGBuilder`.
        """
        # NOTE: Ensure both reps carry a source path for better errors
        if pdb_representation.source is None:
            pdb_representation.source = pdb_source

        builder = CGBuilder(
            mappings,
            accept_alternate_location_codes=accept_alternate_location_codes,
            first_model_only=first_model_only,
        )
        beads = builder.build_beads(pdb_representation, pdb_source=pdb_source)
        cg_representation = cls(cast(list[PDBAtom], beads))
        cg_representation.source = pdb_source

        # Validate that residue sequences match the source (gap-aware)
        if check_sequences and not cg_representation._check_sequences_match(pdb_representation):
            raise CGError(f"CG sequence does not match source sequence for PDB {pdb_source}")

        # Validate that each bead stays close to atoms of its source residue
        if check_geometry:
            cg_representation._check_bead_distances(pdb_representation)

        return cg_representation

    @property
    def beads(self) -> list[CGBead]:
        return cast(list[CGBead], self.atoms)

    def to_csv(self, save: str | Path) -> None:
        """
        Write CG beads to a CSV file.

        The CSV header is the list of all `CGBead` dataclass fields (including fields
        inherited from `PDBAtom`), and each bead is written by dumping these fields in
        that order.

        >>> import tempfile
        >>> from dataclasses import fields as dataclass_fields
        >>> bead = CGBead(
        ...     record=PDBRecord.ATOM,
        ...     name="BB",
        ...     residue_name="GLY",
        ...     residue_id=1,
        ...     x=1.0,
        ...     y=2.0,
        ...     z=3.0,
        ...     chain_id="A",
        ...     bead_type="BB",
        ...     n_atoms_used=1,
        ...     n_atoms_expected=1,
        ... )
        >>> cg = CGRepresentation([bead])
        >>> with tempfile.TemporaryDirectory() as d:
        ...     out = Path(d) / "cg.csv"
        ...     cg.to_csv(out)
        ...     header, row = out.read_text(encoding="utf-8").splitlines()
        >>> header.split(",")
        ['record', 'name', 'residue_name', 'residue_id', 'x', 'y', 'z', ...]
        >>> row.split(",")[:7]
        ['ATOM', 'BB', 'GLY', '1', '1.0', '2.0', '3.0']
        """
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self.beads)
        df.to_csv(save, index=False, header=True)

    def _check_sequences_match(self, pdb_representation: PDBRepresentation) -> bool:
        """
        Return True if this CG representation covers the same residue sequence as source. Gap-aware.
        """

        def names_by_chain_pos(
            rep: PDBRepresentation,
        ) -> dict[str | None, dict[tuple[int, str], str]]:
            def norm_icode(insertion_code: str | None) -> str:
                return (insertion_code or "").strip().upper()

            names: dict[str | None, dict[tuple[int, str], str]] = {}
            for atom in rep.atoms:
                if atom.residue_id is None:
                    raise CGError(f"Atom with missing residue_id in {rep.source!r}: {atom!r}")

                chain_id = atom.chain_id
                resid = int(atom.residue_id)
                pos = (resid, norm_icode(atom.insertion_code))
                by_pos = names.setdefault(chain_id, {})
                if (existing := by_pos.get(pos)) is None:
                    by_pos[pos] = atom.residue_name
                    continue
                if existing != atom.residue_name:
                    position = ResidueKey.from_atom(atom).format_position()
                    raise CGError(
                        f"Conflicting residue names for {position}: {existing!r} vs {atom.residue_name!r}"
                    )
            return names

        source_by_chain_pos = names_by_chain_pos(pdb_representation)
        cg_by_chain_pos = names_by_chain_pos(self)

        # NOTE: Source is the reference; CG may omit residues (unmapped => wildcard gaps)
        for chain_id, source_names in source_by_chain_pos.items():
            cg_names = cg_by_chain_pos.get(chain_id, {})

            # CG must not have residues that do not exist in source
            extra = sorted(set(cg_names) - set(source_names))
            if extra:
                extra_str = ", ".join(str(r) for r in extra[:10])
                more = "" if len(extra) <= 10 else f", ... (+{len(extra) - 10} more)"
                raise CGError(
                    f"CG contains residues not present in source for chain {chain_id or ' '}: "
                    f"{extra_str}{more} in {self.source!r}"
                )

            for (resid, icode), source_residue in sorted(source_names.items()):
                cg_residue = cg_names.get((resid, icode))

                # NOTE: Gap in CG matches any source residue (e.g. unmapped PTMs like TPO)
                if cg_residue is None:
                    continue

                if cg_residue != source_residue:
                    pos_str = f"{resid}{icode}" if icode else f"{resid}"
                    position = f"chain {chain_id or ' '}, resid {pos_str}"
                    raise CGError(
                        f"Sequence mismatch at {position}: "
                        f"CG has {cg_residue!r} vs source {source_residue!r} in {self.source!r}"
                    )

        return True

    def _check_bead_distances(
        self, pdb_representation: PDBRepresentation, distance_threshold: float = 2.0
    ) -> bool:
        """
        Return True if each bead is close to its source residue atoms.
        """
        if distance_threshold <= 0:
            raise ValueError(f"distance_threshold must be > 0, got {distance_threshold!r}")

        def build_source_coords_by_residue() -> dict[ResidueKey, np.ndarray]:
            coords: dict[ResidueKey, list[tuple[float, float, float]]] = {}
            for atom in pdb_representation.atoms:
                coords.setdefault(ResidueKey.from_atom(atom), []).append(
                    (float(atom.x), float(atom.y), float(atom.z))
                )
            return {key: np.asarray(points, dtype=float) for key, points in coords.items()}

        coords_by_residue = build_source_coords_by_residue()

        threshold_sq = float(distance_threshold) ** 2

        for bead in self.beads:
            bead_residue_key = ResidueKey.from_atom(bead)

            source_coords = coords_by_residue.get(bead_residue_key)
            if source_coords is None or source_coords.size == 0:
                raise CGError(
                    f"No source atoms found for residue {bead_residue_key} (bead {bead.name!r}) "
                    f"in {pdb_representation.source}"
                )

            bead_coords = np.asarray((float(bead.x), float(bead.y), float(bead.z)), dtype=float)
            distances_sq = np.sum((source_coords - bead_coords) ** 2, axis=1)
            min_dist_sq = float(distances_sq.min())
            if min_dist_sq > threshold_sq:
                min_dist = float(np.sqrt(min_dist_sq))
                raise CGError(
                    f"Bead too far from source residue atoms at {bead_residue_key}: "
                    f"bead {bead.name!r} min distance {min_dist:.3f} Å "
                    f"> threshold {float(distance_threshold):.3f} Å in {pdb_representation.source}"
                )

        return True
