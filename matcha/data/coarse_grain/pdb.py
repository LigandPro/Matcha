from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from itertools import groupby
from pathlib import Path
from typing import Self, overload

import periodictable


class PDBRecord(StrEnum):
    ATOM = "ATOM"
    HETATM = "HETATM"
    MODEL = "MODEL"
    ENDMDL = "ENDMDL"
    TER = "TER"
    END = "END"


class PDBAtomError(Exception):
    def __init__(self, msg: str):
        self.msg = msg
        super().__init__(msg)


@dataclass(frozen=True, slots=True)
class ResidueKey:
    chain_id: str | None
    residue_id: int
    insertion_code: str | None
    residue_name: str

    @classmethod
    def from_atom(cls, atom: "PDBAtom") -> "ResidueKey":
        return cls(
            chain_id=atom.chain_id,
            residue_id=int(atom.residue_id),
            residue_name=atom.residue_name,
            insertion_code=atom.insertion_code,
        )

    def __str__(self) -> str:
        residue_name = (self.residue_name or "").strip().upper()
        return f"{self.format_position()}-{residue_name}"

    def format_position(self) -> str:
        """Format residue position without residue name (e.g. 'A:23A')."""
        chain = ((self.chain_id or "").strip().upper()) or "_"
        icode = (self.insertion_code or "").strip().upper()
        resid = f"{self.residue_id}{icode}" if icode else f"{self.residue_id}"
        return f"{chain}:{resid}"


@dataclass(frozen=True)
class PDBAtom:
    record: PDBRecord
    name: str
    residue_name: str
    residue_id: int
    x: float
    y: float
    z: float
    chain_id: str | None = None
    alternative_location_code: str | None = None
    insertion_code: str | None = None
    occupancy: float | None = None
    b_factor: float | None = None
    element: str | None = None
    line_no: int | None = None

    @staticmethod
    @overload
    def _field(line: str, start: int, end: int, *, msg: str) -> str: ...

    @staticmethod
    @overload
    def _field(line: str, start: int, end: int, *, msg: None = None) -> str | None: ...

    @staticmethod
    @overload
    def _field[T](line: str, start: int, end: int, type_: Callable[[str], T], *, msg: str) -> T: ...

    @staticmethod
    @overload
    def _field[T](
        line: str, start: int, end: int, type_: Callable[[str], T], *, msg: None = None
    ) -> T | None: ...

    @staticmethod
    def _field[T](
        line: str,
        start: int,
        end: int,
        type_: Callable[[str], T] = str,  # type: ignore[assignment]
        *,
        msg: str | None = None,
    ) -> T | None:
        """
        Return stripped fixed-width PDB field or None if blank/too short.

        If `msg` is provided, raise `PDBAtomError(msg)` instead of returning `None`.
        """
        if start >= len(line):
            if msg is not None:
                raise PDBAtomError(msg)
            return None
        raw = line[start:end].strip()
        if not raw:
            if msg is not None:
                raise PDBAtomError(msg)
            return None
        try:
            return type_(raw)
        except ValueError:
            if msg is not None:
                raise PDBAtomError(msg)
            return None

    @classmethod
    def from_pdb_line(cls, line: str, line_no: int | None = None) -> Self | None:
        """
        Parse a PDB ATOM/HETATM line into an Atom instance.

        >>> line = "ATOM      1  N   ASP A   1      11.104  13.207   2.100  1.00 20.00           N"

        >>> atom = PDBAtom.from_pdb_line(line)
        >>> atom.record, atom.name, atom.residue_name, atom.chain_id, atom.residue_id
        (<PDBRecord.ATOM: 'ATOM'>, 'N', 'ASP', 'A', 1)
        >>> atom.alternative_location_code is None
        True

        >>> atom.residue_name, atom.chain_id, atom.residue_id
        ('ASP', 'A', 1)

        >>> atom.mass
        14.007

        >>> atom.x, atom.y, atom.z
        (11.104, 13.207, 2.1)

        >>> atom.occupancy, atom.b_factor
        (1.0, 20.0)

        Coordinates can legitimately be 0.0 (including -0.0) and should not fail parsing:

        >>> line = "ATOM   4316  HE  ARG C  61       0.180  -0.000  55.077  1.00  0.00           H"
        >>> PDBAtom.from_pdb_line(line).y
        -0.0
        """

        # Parse PDB record type
        record = cls._field(line, 0, 6, msg="Could not parse PDB record type")
        if record not in {PDBRecord.ATOM, PDBRecord.HETATM}:
            raise PDBAtomError(f"Unknown PDB record type: {record}")

        # Parse rest fields
        return cls(
            record=PDBRecord(record),
            name=(an := cls._field(line, 12, 16, msg="Could not parse atom name")),
            alternative_location_code=cls._field(line, 16, 17),
            residue_name=cls._field(line, 17, 20, msg="Could not parse residue name"),
            chain_id=cls._field(line, 21, 22),
            residue_id=cls._field(line, 22, 26, int, msg="Could not parse residue ID"),
            insertion_code=cls._field(line, 26, 27),
            x=cls._field(line, 30, 38, float, msg="Could not parse x coordinate"),
            y=cls._field(line, 38, 46, float, msg="Could not parse y coordinate"),
            z=cls._field(line, 46, 54, float, msg="Could not parse z coordinate"),
            occupancy=cls._field(line, 54, 60, float),
            b_factor=cls._field(line, 60, 66, float),
            element=cls._field(line, 76, 78) or cls._element_from_atom_name(an or ""),
            line_no=line_no,
        )

    @property
    def mass(self) -> float | None:
        """
        Atomic mass inferred from `element`. Returns `None` when the element is unknown.

        >>> line = "ATOM      1  N   ASP A   1      11.104  13.207   2.100  1.00 20.00           N"
        >>> atom = PDBAtom.from_pdb_line(line)
        >>> atom.mass
        14.007...
        """
        try:
            return periodictable.elements.symbol(self.element.title()).mass  # type: ignore
        except (ValueError, AttributeError):
            return None

    def to_line(self, serial: int) -> str:
        """
        Format this atom as a strict fixed-width PDB ATOM/HETATM line.

        Fields that are `None` are written as blanks (e.g. occupancy, B-factor, element).

        >>> atom = PDBAtom(
        ...     record=PDBRecord.ATOM,
        ...     name="BB",
        ...     residue_name="GLY",
        ...     chain_id="A",
        ...     residue_id=1,
        ...     x=1.0,
        ...     y=2.0,
        ...     z=3.0,
        ... )
        >>> line = atom.to_line(1)
        >>> line.startswith("ATOM")
        True
        >>> (line[54:60], line[60:66], line[76:78])  # occupancy, B-factor, element fields
        ('      ', '      ', '  ')
        >>> line[30:38].strip(), line[38:46].strip(), line[46:54].strip()
        ('1.000', '2.000', '3.000')
        """
        if self.record not in (PDBRecord.ATOM, PDBRecord.HETATM):
            raise PDBAtomError(f"Cannot format PDB line for record {self.record!r}")

        record = f"{self.record.value:<6s}"
        atom_name = (self.name or "")[:4]
        altloc = ((self.alternative_location_code or "")[:1]).strip()
        resname = (self.residue_name or "")[:3]
        chain = (self.chain_id or "")[:1]

        resid = f"{int(self.residue_id):4d}" if self.residue_id is not None else "    "
        icode = (self.insertion_code or "")[:1]

        x = f"{float(self.x):8.3f}"
        y = f"{float(self.y):8.3f}"
        z = f"{float(self.z):8.3f}"

        occ = f"{float(self.occupancy):6.2f}" if self.occupancy is not None else "      "
        bfac = f"{float(self.b_factor):6.2f}" if self.b_factor is not None else "      "

        elem = (self.element or "").strip()
        elem = elem.title() if elem else ""
        elem_field = f"{elem:>2s}" if elem else "  "

        # PDB columns (1-based):
        #  1-6  record, 7-11 serial, 13-16 atom, 17 altLoc, 18-20 resName, 22 chain,
        # 23-26 resSeq, 27 iCode, 31-54 xyz, 55-60 occ, 61-66 temp, 77-78 element.
        return (
            f"{record}{serial:5d} {atom_name:>4s}{altloc:1s}{resname:>3s} {chain:1s}"
            f"{resid}{icode:1s}   {x}{y}{z}{occ}{bfac}          {elem_field:>2s}  \n"
        )

    @staticmethod
    def _element_from_atom_name(atom_name: str, *, greedy_hydrogen: bool = True) -> str:
        """
        Determine the element symbol from atom name and element field.

        >>> PDBAtom._element_from_atom_name(" CA ")
        'C'

        >>> PDBAtom._element_from_atom_name("1HD2")
        'H'

        >>> PDBAtom._element_from_atom_name(" FE ")
        'Fe'

        >>> PDBAtom._element_from_atom_name(" OXT ")
        'O'

        >>> PDBAtom._element_from_atom_name("    ")
        'C'

        >>> PDBAtom._element_from_atom_name("HG1")
        'H'

        >>> PDBAtom._element_from_atom_name("HG1", greedy_hydrogen=False)
        'Hg'
        """

        name = (atom_name or "").strip().lstrip("0123456789")
        if not name:
            return "C"

        # Special case: Alpha Carbon
        if name.upper() == "CA":
            return "C"

        # By default, treat any atom name that starts with "H" (after stripping leading
        # digits like "1HD2") as Hydrogen. This avoids misclassifying common protein
        # hydrogen names like "HG1" as Mercury ("Hg") when the element field is blank.
        if greedy_hydrogen and name and name[0].upper() == "H":
            return "H"

        # Try to match 2-letter element (e.g. "Fe", "Mg")
        # Only if the original name didn't start with a digit (which usually implies Hydrogen)
        if not name[0].isdigit() and len(name) >= 2:
            candidate = name[:2].title()
            try:
                periodictable.elements.symbol(candidate)
                return candidate
            except (ValueError, AttributeError):
                pass

        # Fallback to first letter
        return name[0].upper()


@dataclass
class PDBRepresentation:
    """
    Container for a molecule represented as a list of PDBAtom-compatible records.
    """

    atoms: list[PDBAtom]
    source: Path | None = None

    @staticmethod
    def group_atoms_by_residue(
        pdb_representation: "PDBRepresentation",
    ) -> Iterator[tuple[ResidueKey, list[PDBAtom]]]:
        # NOTE: `groupby` groups *consecutive* elements. PDB atoms are read in file order,
        # and atoms belonging to a residue are expected to be contiguous in typical PDBs
        for residue_key, atom_group in groupby(pdb_representation.atoms, key=ResidueKey.from_atom):
            yield residue_key, list(atom_group)

    @classmethod
    def from_pdb(
        cls,
        pdb_path: str | Path,
        *,
        first_model_only: bool = True,
        accept_altloc: tuple[str, ...] = ("", "A"),
    ) -> "PDBRepresentation":
        pdb_path = Path(pdb_path)
        accept = tuple((a or "").strip().upper() for a in accept_altloc)

        atoms: list[PDBAtom] = []
        model_index = 0

        with pdb_path.open("r", encoding="utf-8", errors="replace") as pdb_file:
            for line_no, line in enumerate(pdb_file, start=1):
                record = line[:6].strip()
                match record:
                    case PDBRecord.MODEL:
                        model_index += 1
                        if first_model_only and model_index > 1:
                            break
                        continue

                    case PDBRecord.ENDMDL | PDBRecord.END:
                        if first_model_only:
                            break
                        continue

                    case PDBRecord.TER:
                        continue

                    case PDBRecord.ATOM | PDBRecord.HETATM:
                        try:
                            atom = PDBAtom.from_pdb_line(line, line_no)
                        except PDBAtomError as e:
                            raise PDBAtomError(
                                f"Failed to parse line in PDB {pdb_path}:{line_no} - {e.msg}"
                            ) from e

                        if atom is None:
                            continue
                        if (atom.alternative_location_code or "").strip().upper() not in accept:
                            continue
                        atoms.append(atom)

                    case _:
                        continue

        return cls(atoms=atoms)

    def to_pdb(self, save: str | Path) -> str:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)

        with save.open("w", encoding="utf-8") as f:
            for i, atom in enumerate(self.atoms, start=1):
                f.write(atom.to_line(i))
            f.write("END\n")

        return str(save)

    def sequences(self) -> dict[str | None, list[str | None]]:
        """
        Return sequences by chain accounting for gaps.

        Residues are ordered by `(residue_id, insertion_code)` where the insertion code is treated
        as an uppercase alphanumeric string and blanks sort first (i.e. 10 < 10A < 10B < 11).
        Missing residue IDs produce `None` gaps. Conflicting residue names for the same
        `(chain_id, residue_id, insertion_code)` key raise a ValueError.

        >>> mk = lambda res, chain, resid, icode=None: PDBAtom(
        ...     record=PDBRecord.ATOM,
        ...     name="CA",
        ...     residue_name=res,
        ...     chain_id=chain,
        ...     residue_id=resid,
        ...     insertion_code=icode,
        ...     x=0.0,
        ...     y=0.0,
        ...     z=0.0,
        ... )
        >>> rep = PDBRepresentation(atoms=[mk("ALA", "X", 1), mk("GLY", "X", 3), mk("SER", "Y", 1)])
        >>> rep.sequences()
        {'X': ['ALA', None, 'GLY'], 'Y': ['SER']}

        Insertion codes are included in alphanumeric order:

        >>> rep = PDBRepresentation(
        ...     atoms=[mk("ALA", "X", 1), mk("GLY", "X", 1, "A"), mk("SER", "X", 2)]
        ... )
        >>> rep.sequences()
        {'X': ['ALA', 'GLY', 'SER']}

        Conflicts for the same residue key error:

        >>> rep = PDBRepresentation(atoms=[mk("ALA", "X", 1), mk("GLY", "X", 1)])
        >>> rep.sequences()  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Conflicting residue names for X:1: 'ALA' vs 'GLY'
        """

        def norm_icode(insertion_code: str | None) -> str:
            return (insertion_code or "").strip().upper()

        # Track residue names per chain/(resid, icode)
        names_by_chain_pos: dict[str | None, dict[tuple[int, str], str]] = {}

        for atom in self.atoms:
            if atom.residue_id is None:
                raise ValueError(f"Atom with missing residue_id: {atom!r}")

            chain_id = atom.chain_id
            resid = int(atom.residue_id)
            pos = (resid, norm_icode(atom.insertion_code))
            names_by_chain_pos.setdefault(chain_id, {})

            existing = names_by_chain_pos[chain_id].get(pos)
            if existing is None:
                names_by_chain_pos[chain_id][pos] = atom.residue_name
                continue
            if existing != atom.residue_name:
                position = ResidueKey.from_atom(atom).format_position()
                raise ValueError(
                    f"Conflicting residue names for {position}: {existing!r} vs {atom.residue_name!r}"
                )

        # Build gap-aware sequences for each chain (including insertion codes)
        sequences: dict[str | None, list[str | None]] = {}
        for chain_id, names_by_pos in names_by_chain_pos.items():
            if not names_by_pos:
                sequences[chain_id] = []
                continue

            residue_ids = {resid for (resid, _icode) in names_by_pos.keys()}
            min_resid = min(residue_ids)
            max_resid = max(residue_ids)

            seq: list[str | None] = []
            for resid in range(min_resid, max_resid + 1):
                positions = sorted(
                    ((r, icode) for (r, icode) in names_by_pos.keys() if r == resid),
                    key=lambda p: p[1],
                )
                if not positions:
                    seq.append(None)
                    continue
                for pos in positions:
                    seq.append(names_by_pos[pos])

            sequences[chain_id] = seq

        return sequences
