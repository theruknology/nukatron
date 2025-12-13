"""
Nukatron Loader Module: Robust PDB Ingestor
============================================
Loads and intelligently parses PDB files using MDAnalysis.
Automatically separates atoms into four groups:
  1. Ligand (Drug/Small Molecule)
  2. Protein (Standard Amino Acids)
  3. DNA (Nucleic Acids)
  4. Water (Solvent Molecules)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import MDAnalysis as mda
from MDAnalysis.coordinates.memory import MemoryReader
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


class PDBLoader:
    """
    Robust PDB loader with intelligent atom group separation.
    
    Attributes:
        u (MDAnalysis.Universe): The loaded molecular system
        ligand_atoms (MDAnalysis.AtomGroup): Selected ligand atoms
        protein_atoms (MDAnalysis.AtomGroup): Selected protein atoms
        dna_atoms (MDAnalysis.AtomGroup): Selected DNA atoms
        water_atoms (MDAnalysis.AtomGroup): Selected water molecules
    """
    
    # Standard nucleotide resnames
    NUCLEOTIDE_RESNAMES = {'A', 'G', 'C', 'T', 'U', 'DA', 'DG', 'DC', 'DT'}
    
    # Standard water resnames
    WATER_RESNAMES = {'HOH', 'WAT', 'TIP3', 'TIP4', 'SPC', 'OHO'}
    
    # Standard amino acid resnames
    PROTEIN_RESNAMES = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }
    
    def __init__(self, pdb_path: str, ligand_resnames: Optional[list] = None):
        """
        Initialize the PDB loader.
        
        Args:
            pdb_path (str): Path to PDB file
            ligand_resnames (list, optional): Custom ligand residue names.
                                               If None, auto-detects non-standard residues.
        
        Raises:
            FileNotFoundError: If PDB file does not exist
            IOError: If MDAnalysis fails to parse the file
        """
        self.pdb_path = Path(pdb_path)
        self.ligand_resnames = ligand_resnames or []
        self.u = None
        self.ligand_atoms = None
        self.protein_atoms = None
        self.dna_atoms = None
        self.water_atoms = None
        
        if not self.pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")
        
        self._load_pdb()
        self._classify_atoms()
    
    def _load_pdb(self) -> None:
        """Load PDB using MDAnalysis with robust error handling."""
        try:
            self.u = mda.Universe(str(self.pdb_path))
            logger.info(f"Loaded PDB: {self.pdb_path.name} with {self.u.atoms.n_atoms} atoms")
        except Exception as e:
            logger.error(f"Failed to load PDB: {e}")
            raise IOError(f"MDAnalysis could not parse {self.pdb_path}: {e}")
    
    def _classify_atoms(self) -> None:
        """
        Intelligently classify atoms into four groups.
        
        Classification priority:
        1. Water (by resname)
        2. DNA (by resname or nucleic selection)
        3. Protein (by resname)
        4. Ligand (remaining non-standard residues or specified resnames)
        """
        try:
            # Extract all residue names in the system
            all_resnames = set(self.u.atoms.resnames)
            
            # 1. Select Water molecules
            water_selection = ' or '.join([f'resname {res}' for res in self.WATER_RESNAMES])
            try:
                self.water_atoms = self.u.select_atoms(water_selection)
            except:
                self.water_atoms = self.u.atoms[0:0]  # Empty selection
            
            logger.info(f"Water molecules: {self.water_atoms.n_atoms} atoms")
            
            # 2. Select DNA/RNA
            try:
                # MDAnalysis nucleic selection
                self.dna_atoms = self.u.select_atoms('nucleic')
            except:
                # Fallback: manual selection by resname
                dna_selection = ' or '.join([f'resname {res}' for res in self.NUCLEOTIDE_RESNAMES])
                try:
                    self.dna_atoms = self.u.select_atoms(dna_selection)
                except:
                    self.dna_atoms = self.u.atoms[0:0]  # Empty selection
            
            logger.info(f"DNA/RNA molecules: {self.dna_atoms.n_atoms} atoms")
            
            # 3. Select Protein (standard amino acids)
            protein_selection = ' or '.join([f'resname {res}' for res in self.PROTEIN_RESNAMES])
            try:
                self.protein_atoms = self.u.select_atoms(protein_selection)
            except:
                self.protein_atoms = self.u.atoms[0:0]  # Empty selection
            
            logger.info(f"Protein residues: {self.protein_atoms.n_atoms} atoms")
            
            # 4. Select Ligand
            # Priority: explicit ligand_resnames > remaining non-standard residues
            used_atoms = set(self.water_atoms.indices) | set(self.dna_atoms.indices) | set(self.protein_atoms.indices)
            remaining_atoms = self.u.atoms[~np.isin(self.u.atoms.indices, list(used_atoms))]
            
            if self.ligand_resnames:
                # Use explicitly specified ligand resnames
                ligand_selection = ' or '.join([f'resname {res}' for res in self.ligand_resnames])
                try:
                    self.ligand_atoms = self.u.select_atoms(ligand_selection)
                except:
                    self.ligand_atoms = remaining_atoms
            else:
                # Auto-detect: use remaining non-classified atoms
                self.ligand_atoms = remaining_atoms
            
            logger.info(f"Ligand atoms: {self.ligand_atoms.n_atoms} atoms")
            
            # Sanity check
            total_classified = (self.ligand_atoms.n_atoms + self.protein_atoms.n_atoms + 
                               self.dna_atoms.n_atoms + self.water_atoms.n_atoms)
            if total_classified != self.u.atoms.n_atoms:
                logger.warning(f"Atom count mismatch: classified {total_classified}, "
                              f"expected {self.u.atoms.n_atoms}")
        
        except Exception as e:
            logger.error(f"Error during atom classification: {e}")
            raise
    
    def get_atom_groups(self) -> Dict[str, Any]:
        """
        Return all classified atom groups.
        
        Returns:
            dict: Dictionary with keys 'ligand', 'protein', 'dna', 'water'
        """
        return {
            'ligand': self.ligand_atoms,
            'protein': self.protein_atoms,
            'dna': self.dna_atoms,
            'water': self.water_atoms
        }
    
    def get_coordinates(self) -> Dict[str, np.ndarray]:
        """
        Get current frame coordinates for all atom groups.
        
        Returns:
            dict: Dictionary with keys 'ligand', 'protein', 'dna', 'water'
                  containing (N, 3) coordinate arrays
        """
        return {
            'ligand': self.ligand_atoms.positions,
            'protein': self.protein_atoms.positions,
            'dna': self.dna_atoms.positions,
            'water': self.water_atoms.positions
        }
    
    def get_residue_summary(self) -> pd.DataFrame:
        """
        Generate a summary of all residues in the system.
        
        Returns:
            pd.DataFrame: Columns: [resname, resid, n_atoms, group]
        """
        residues = []
        
        for group_name, atoms in self.get_atom_groups().items():
            if atoms.n_atoms == 0:
                continue
            for residue in atoms.residues:
                residues.append({
                    'resname': residue.resname,
                    'resid': residue.resnum,
                    'n_atoms': residue.atoms.n_atoms,
                    'group': group_name
                })
        
        return pd.DataFrame(residues)
    
    def save_group_pdb(self, output_dir: str, group_name: str = 'ligand') -> str:
        """
        Save a specific atom group to a PDB file.
        
        Args:
            output_dir (str): Directory to save PDB
            group_name (str): One of 'ligand', 'protein', 'dna', 'water'
        
        Returns:
            str: Path to saved PDB file
        """
        if group_name not in self.get_atom_groups():
            raise ValueError(f"Unknown group: {group_name}")
        
        output_path = Path(output_dir) / f"{self.pdb_path.stem}_{group_name}.pdb"
        atoms = self.get_atom_groups()[group_name]
        atoms.write(str(output_path))
        logger.info(f"Saved {group_name} to {output_path}")
        return str(output_path)


def load_pdb(pdb_path: str, ligand_resnames: Optional[list] = None) -> PDBLoader:
    """
    Convenience function to load a PDB file.
    
    Args:
        pdb_path (str): Path to PDB file
        ligand_resnames (list, optional): Custom ligand residue names
    
    Returns:
        PDBLoader: Loader object with classified atom groups
    """
    return PDBLoader(pdb_path, ligand_resnames)


if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python loader.py <pdb_file> [ligand_resname1 ligand_resname2 ...]")
        sys.exit(1)
    
    pdb_file = sys.argv[1]
    ligand_names = sys.argv[2:] if len(sys.argv) > 2 else None
    
    loader = load_pdb(pdb_file, ligand_names)
    print("\n=== Atom Group Summary ===")
    print(loader.get_residue_summary())
