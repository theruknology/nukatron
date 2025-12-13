"""
Nukatron Engine Module: Tri-Body Physics Engine
================================================
Generates comprehensive interaction fingerprints using ProLIF.
Detects interactions between:
  - Ligand ↔ Protein
  - Ligand ↔ DNA
  - Protein ↔ DNA (optional context)
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
import pandas as pd
import MDAnalysis as mda

# ProLIF for interaction fingerprinting
try:
    from prolif.interactions import Interaction
    from prolif.fingerprint import IFPCount
    PROLIF_AVAILABLE = True
except ImportError:
    PROLIF_AVAILABLE = False
    logging.warning("ProLIF not available. Will use MDAnalysis distance-based interactions.")

logger = logging.getLogger(__name__)


class InteractionEngine:
    """
    Tri-Body Interaction Fingerprinter.
    
    Generates structured DataFrames of interactions between three molecular entities:
    Ligand, Protein, and DNA. Uses distance-based criteria for robustness.
    """
    
    # Distance thresholds (Ångströms)
    HBOND_DISTANCE = 3.5      # H-bond acceptor-donor distance
    VDWAAL_DISTANCE = 4.5     # Van der Waals contact
    PI_STACKING_DISTANCE = 5.5  # π-π stacking
    
    # Interaction types
    INTERACTION_TYPES = [
        'HBond', 'VdW', 'PiStacking', 'SaltBridge', 'CationPi'
    ]
    
    def __init__(self, universe: mda.Universe, ligand_atoms, protein_atoms, dna_atoms):
        """
        Initialize the interaction engine.
        
        Args:
            universe (mda.Universe): The loaded molecular system
            ligand_atoms (mda.AtomGroup): Ligand/drug atoms
            protein_atoms (mda.AtomGroup): Protein atoms
            dna_atoms (mda.AtomGroup): DNA atoms
        """
        self.u = universe
        self.ligand = ligand_atoms
        self.protein = protein_atoms
        self.dna = dna_atoms
        
        # Store interaction results
        self.interactions_df = None
        self.interaction_summary = {}
    
    def _compute_distance_interactions(self, group1: mda.AtomGroup, group2: mda.AtomGroup,
                                       threshold: float = 4.5) -> List[Dict]:
        """
        Compute distance-based interactions between two atom groups.
        
        Args:
            group1 (mda.AtomGroup): First atom group
            group2 (mda.AtomGroup): Second atom group
            threshold (float): Distance cutoff in Ångströms
        
        Returns:
            list: List of interaction dictionaries
        """
        interactions = []
        
        if group1.n_atoms == 0 or group2.n_atoms == 0:
            return interactions
        
        # Compute all pairwise distances using numpy
        pos1 = group1.positions
        pos2 = group2.positions
        
        # Calculate pairwise distances
        distances = np.sqrt(((pos1[:, np.newaxis, :] - pos2[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Find close contacts
        close_pairs = np.where(distances < threshold)
        
        for i, j in zip(close_pairs[0], close_pairs[1]):
            atom1 = group1[i]
            atom2 = group2[j]
            distance = distances[i, j]
            
            # Classify interaction type by distance
            interaction_type = self._classify_interaction(distance, atom1, atom2)
            
            interactions.append({
                'atom1_name': atom1.name,
                'atom1_resname': atom1.resname,
                'atom1_resid': atom1.resnum,
                'atom2_name': atom2.name,
                'atom2_resname': atom2.resname,
                'atom2_resid': atom2.resnum,
                'distance': distance,
                'interaction_type': interaction_type
            })
        
        return interactions
    
    def _classify_interaction(self, distance: float, atom1, atom2) -> str:
        """
        Classify interaction type based on distance and atom properties.
        
        Args:
            distance (float): Distance between atoms (Å)
            atom1: First atom
            atom2: Second atom
        
        Returns:
            str: Interaction type label
        """
        # Hydrogen bond
        if distance < self.HBOND_DISTANCE:
            # Check for H-bond donors/acceptors
            hbond_atoms = {'O', 'N', 'S'}
            if atom1.element in hbond_atoms or atom2.element in hbond_atoms:
                return 'HBond'
        
        # Van der Waals contact
        if distance < self.VDWAAL_DISTANCE:
            return 'VdW'
        
        # π-stacking (for aromatic residues)
        if distance < self.PI_STACKING_DISTANCE:
            aromatic_atoms = {'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'}  # Phe, Tyr, Trp
            if atom1.name in aromatic_atoms or atom2.name in aromatic_atoms:
                return 'PiStacking'
        
        return 'Contact'
    
    def compute_ligand_protein_interactions(self) -> pd.DataFrame:
        """
        Compute all Ligand ↔ Protein interactions.
        
        Returns:
            pd.DataFrame: Interaction table with columns:
                         [ligand_res, protein_res, distance, interaction_type]
        """
        logger.info("Computing Ligand-Protein interactions...")
        
        interactions = self._compute_distance_interactions(self.ligand, self.protein)
        
        df = pd.DataFrame(interactions)
        if not df.empty:
            df['pair_type'] = 'Ligand-Protein'
            df.rename(columns={
                'atom1_resname': 'ligand_resname',
                'atom1_resid': 'ligand_resid',
                'atom2_resname': 'protein_resname',
                'atom2_resid': 'protein_resid'
            }, inplace=True)
        
        self.interaction_summary['ligand_protein'] = len(df)
        logger.info(f"Found {len(df)} Ligand-Protein interactions")
        
        return df
    
    def compute_ligand_dna_interactions(self) -> pd.DataFrame:
        """
        Compute all Ligand ↔ DNA interactions.
        
        Returns:
            pd.DataFrame: Interaction table
        """
        logger.info("Computing Ligand-DNA interactions...")
        
        interactions = self._compute_distance_interactions(self.ligand, self.dna)
        
        df = pd.DataFrame(interactions)
        if not df.empty:
            df['pair_type'] = 'Ligand-DNA'
            df.rename(columns={
                'atom1_resname': 'ligand_resname',
                'atom1_resid': 'ligand_resid',
                'atom2_resname': 'dna_resname',
                'atom2_resid': 'dna_resid'
            }, inplace=True)
        
        self.interaction_summary['ligand_dna'] = len(df)
        logger.info(f"Found {len(df)} Ligand-DNA interactions")
        
        return df
    
    def compute_protein_dna_interactions(self) -> pd.DataFrame:
        """
        Compute all Protein ↔ DNA interactions (context).
        
        Returns:
            pd.DataFrame: Interaction table
        """
        logger.info("Computing Protein-DNA interactions...")
        
        interactions = self._compute_distance_interactions(self.protein, self.dna)
        
        df = pd.DataFrame(interactions)
        if not df.empty:
            df['pair_type'] = 'Protein-DNA'
            df.rename(columns={
                'atom1_resname': 'protein_resname',
                'atom1_resid': 'protein_resid',
                'atom2_resname': 'dna_resname',
                'atom2_resid': 'dna_resid'
            }, inplace=True)
        
        self.interaction_summary['protein_dna'] = len(df)
        logger.info(f"Found {len(df)} Protein-DNA interactions")
        
        return df
    
    def compute_all_interactions(self) -> pd.DataFrame:
        """
        Compute all three-body interactions and merge into a single DataFrame.
        
        Returns:
            pd.DataFrame: Combined interaction table
        """
        logger.info("=" * 60)
        logger.info("Computing Tri-Body Interaction Fingerprint")
        logger.info("=" * 60)
        
        # Compute each pair
        lig_prot = self.compute_ligand_protein_interactions()
        lig_dna = self.compute_ligand_dna_interactions()
        prot_dna = self.compute_protein_dna_interactions()
        
        # Combine all
        all_interactions = pd.concat([lig_prot, lig_dna, prot_dna], 
                                     ignore_index=True, sort=False)
        
        self.interactions_df = all_interactions
        
        logger.info("-" * 60)
        logger.info(f"Total interactions found: {len(all_interactions)}")
        logger.info(f"Interaction type distribution:")
        for itype, count in all_interactions['interaction_type'].value_counts().items():
            logger.info(f"  {itype}: {count}")
        logger.info("=" * 60)
        
        return all_interactions
    
    def get_interaction_fingerprint(self, normalize: bool = False) -> Dict[str, int]:
        """
        Generate a simple interaction fingerprint as a count dictionary.
        
        Args:
            normalize (bool): If True, normalize by residue count
        
        Returns:
            dict: Count of each interaction type
        """
        if self.interactions_df is None:
            self.compute_all_interactions()
        
        fingerprint = self.interactions_df['interaction_type'].value_counts().to_dict()
        
        if normalize:
            total = sum(fingerprint.values())
            if total > 0:
                fingerprint = {k: v/total for k, v in fingerprint.items()}
        
        return fingerprint
    
    def save_interactions(self, output_path: str) -> str:
        """
        Save interaction table to CSV.
        
        Args:
            output_path (str): Path to output CSV file
        
        Returns:
            str: Path to saved file
        """
        if self.interactions_df is None:
            self.compute_all_interactions()
        
        self.interactions_df.to_csv(output_path, index=False)
        logger.info(f"Saved interactions to {output_path}")
        return output_path


def create_engine(universe: mda.Universe, atom_groups: Dict[str, mda.AtomGroup]) -> InteractionEngine:
    """
    Convenience function to create an interaction engine.
    
    Args:
        universe (mda.Universe): The loaded molecular system
        atom_groups (dict): Output from loader.get_atom_groups()
    
    Returns:
        InteractionEngine: Initialized engine
    """
    return InteractionEngine(
        universe,
        atom_groups['ligand'],
        atom_groups['protein'],
        atom_groups['dna']
    )


if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python engine.py <pdb_file>")
        sys.exit(1)
    
    from loader import load_pdb
    
    pdb_file = sys.argv[1]
    loader = load_pdb(pdb_file)
    
    engine = create_engine(loader.u, loader.get_atom_groups())
    interactions_df = engine.compute_all_interactions()
    
    print("\n=== Interaction Summary ===")
    print(interactions_df.groupby('interaction_type').size())
