"""
Parser for ITP-like Martini mapping and force-field files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

type ResName = str  # residue name like "ALA", "GLY", etc.
type BeadDummyName = str  # dummy names like BB (backbone), SC1 (sidechain 1), etc.
type BeadType = str  # true bead types like P1, C1, etc.
type AtomName = str  # atom name like "CA", "CB", etc.
type PathLike = str | Path
type Table = list[list[str]]


class MartiniParseError(ValueError):
    """
    Raised when an input file cannot be parsed under the requested strictness.
    """

    pass


@dataclass
class Section:
    """
    Represents a section in an ITP file, containing a name and a table of data.
    """

    name: str
    table: Table = field(default_factory=list)

    def __iter__(self):
        return iter(self.table)


@dataclass
class MoleculeNameSection(Section):
    """
    Section containing molecule name (e.g. [moleculetype] or [molecule]).
    """

    @property
    def resname(self) -> str | None:
        if self.table and self.table[0]:
            return self.table[0][0]
        return None


@dataclass
class FFAtomsSection(Section):
    """
    [atoms] section in force-field files.
    Columns: nr type resnr residue atom cgnr charge mass
    """

    def iter_beads(self) -> Iterator[tuple[str, str]]:
        """Yields (bead_name, bead_type)."""
        for row in self.table:
            if len(row) >= 5:
                yield row[4], row[1]


@dataclass
class MapAtomsSection(Section):
    """
    [atoms] section in mapping files.
    Columns: nr atom bead
    """

    def iter_mapping(self) -> Iterator[tuple[str, str]]:
        """Yields (atom_name, bead_name)."""
        for row in self.table:
            if len(row) >= 3:
                yield row[1], row[2]


@dataclass
class ITPParser:
    """
    Generic parser for ITP-like files.
    Produces a structured representation of sections and handles macro substitution.
    """

    path: Path
    section_types: dict[str, type[Section]] = field(default_factory=dict)
    macros: dict[str, str] = field(default_factory=dict)
    sections: list[Section] = field(default_factory=list)

    def parse(self) -> ITPParser:
        current_section_name = None
        buffer: Table = []

        with self.path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = self._strip_comment(line).strip()
                if not line:
                    continue

                if line.startswith("[") and line.endswith("]"):
                    if current_section_name:
                        self._add_section(current_section_name, buffer)
                    current_section_name = line[1:-1].strip().lower()
                    buffer = []
                    continue

                if current_section_name in ("macros", "variables"):
                    self._parse_macro(current_section_name, line)
                    continue

                # Tokenize and substitute
                tokens = line.split()
                tokens = [self._substitute(t) for t in tokens]
                buffer.append(tokens)

        if current_section_name:
            self._add_section(current_section_name, buffer)
        return self

    def _add_section(self, name: str, table: Table):
        cls = self.section_types.get(name, Section)
        self.sections.append(cls(name, table))

    def _strip_comment(self, line: str) -> str:
        for marker in (";", "#"):
            idx = line.find(marker)
            if idx != -1:
                line = line[:idx]
        return line

    def _parse_macro(self, section: str, line: str):
        parts = line.split(None, 1)
        if len(parts) < 2:
            return
        key, val = parts[0], parts[1].strip().strip('"')
        if section == "macros":
            self.macros[key] = val
        else:
            self.macros.setdefault(key, val)

    def _substitute(self, token: str) -> str:
        if token.startswith("$"):
            return self.macros.get(token[1:], token)
        return token


@dataclass(slots=True)
class MartiniItpParser:
    """
    Stateful parser for Martini mapping and force-field files.
    """

    strict: bool = True

    @classmethod
    def parse(
        cls, map_path: PathLike, ff_path: PathLike
    ) -> dict[ResName, dict[BeadType, set[AtomName]]]:
        """
        Merge a map file and a force-field file for a single residue.

        The map determines which atoms belong to each *bead name*.
        The force field determines which *bead type* corresponds to each bead name.
        The result is grouped by bead type.
        """

        # Convert paths
        map_path = Path(map_path)
        ff_path = Path(ff_path)

        # Parse force field and map
        ff_dict = cls.parse_ff(ff_path)
        resname, bead_dummyname_to_atomnames_map = cls.parse_map(map_path)

        # Check that residue exists in force field
        if resname not in ff_dict:
            raise KeyError(f"Residue '{resname}' not found in force field: {ff_path}. ")

        # Merge map and force field by bead type
        bead_dummyname_to_type_map = ff_dict[resname]
        accumulator: dict[ResName, dict[BeadType, set[AtomName]]] = {resname: {}}

        # Iterate over dummy bead names in map and convert them to bead types
        for bead_dummyname, atoms in bead_dummyname_to_atomnames_map.items():
            bead_type = bead_dummyname_to_type_map.get(bead_dummyname)
            if bead_type is None:
                if cls.strict:
                    raise KeyError(
                        f"Bead '{bead_dummyname}' in map '{map_path}' not found for residue '{resname}' "
                    )
                continue
            accumulator[resname].setdefault(bead_type, set()).update(atoms)

        return accumulator

    @classmethod
    def parse_ff(cls, ff_path: PathLike) -> dict[ResName, dict[BeadDummyName, BeadType]]:
        """
        Parse a Martini ``*.ff``-like file.
        """
        section_types = {
            "moleculetype": MoleculeNameSection,
            "atoms": FFAtomsSection,
        }
        itp = ITPParser(Path(ff_path), section_types=section_types).parse()
        residue_bead_mappings: dict[ResName, dict[BeadDummyName, BeadType]] = {}
        current_resname: ResName | None = None

        for section in itp.sections:
            match section:
                case MoleculeNameSection(resname=resname) if resname:
                    current_resname = resname
                    residue_bead_mappings.setdefault(current_resname, {})

                case FFAtomsSection() as atoms:
                    if current_resname is None:
                        if cls.strict:
                            raise MartiniParseError(
                                f"{ff_path}: [atoms] section before any [moleculetype]"
                            )
                        continue

                    for bead_name, bead_type in atoms.iter_beads():
                        residue_bead_mappings[current_resname][bead_name] = bead_type

        return residue_bead_mappings

    @classmethod
    def parse_map(cls, map_path: PathLike) -> tuple[ResName, dict[BeadDummyName, set[AtomName]]]:
        """
        Parse a vermouth ``*.map`` file for a single molecule/residue.
        """
        section_types = {
            "molecule": MoleculeNameSection,
            "atoms": MapAtomsSection,
        }
        itp = ITPParser(Path(map_path), section_types=section_types).parse()
        resname: ResName | None = None
        bead_name_to_atoms: dict[BeadDummyName, set[AtomName]] = {}

        for section in itp.sections:
            match section:
                case MoleculeNameSection(resname=r) if r and resname is None:
                    resname = r

                case MapAtomsSection() as atoms:
                    for atom_name, bead_name in atoms.iter_mapping():
                        if bead_name.startswith("!"):
                            continue
                        bead_name_to_atoms.setdefault(bead_name, set()).add(atom_name)

        if not resname:
            raise MartiniParseError(f"{map_path}: no [molecule] section found")
        return resname, bead_name_to_atoms
