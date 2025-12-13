"""
Nukatron Water Bridge Module: The "ETS Special"
================================================
ETS Transcription Factors rely heavily on coordinated water molecules for
DNA recognition and binding specificity. This module identifies water bridges
that connect different molecular entities.

A Water Bridge is defined as:
  Water ← (< 3.5 Å) → Entity A AND Water ← (< 3.5 Å) → Entity B
  where Entity A ≠ Entity B (different molecular groups)
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import MDAnalysis as mda

logger = logging.getLogger(__name__)


class WaterBridgeDetector:
    """
    Detects water-mediated interactions in ETS factor complexes.
    
    Water bridges are critical for ETS-DNA recognition and are often missed
    by traditional interaction analysis.
    """
    
    # Water-to-atom distance threshold (Ångströms)
    WATER_BRIDGE_DISTANCE = 3.5
    
    def __init__(self, universe: mda.Universe, water_atoms: mda.AtomGroup,
                 protein_atoms: mda.AtomGroup, dna_atoms: mda.AtomGroup,
                 ligand_atoms: Optional[mda.AtomGroup] = None):
        """
        Initialize the water bridge detector.
        
        Args:
            universe (mda.Universe): The loaded molecular system
            water_atoms (mda.AtomGroup): Water molecules
            protein_atoms (mda.AtomGroup): Protein atoms
            dna_atoms (mda.AtomGroup): DNA atoms
            ligand_atoms (mda.AtomGroup, optional): Ligand atoms
        """
        self.u = universe
        self.water = water_atoms
        self.protein = protein_atoms
        self.dna = dna_atoms
        self.ligand = ligand_atoms
        
        self.water_bridges_df = None
        self.bridge_summary = {}
    
    def _find_neighbors(self, query_atoms: mda.AtomGroup, target_atoms: mda.AtomGroup,
                        threshold: float = WATER_BRIDGE_DISTANCE) -> Dict[int, List[int]]:
        """
        Find neighbors within distance threshold.
        
        Args:
            query_atoms (mda.AtomGroup): Atoms to query from
            target_atoms (mda.AtomGroup): Atoms to query to
            threshold (float): Distance cutoff (Å)
        
        Returns:
            dict: Mapping {query_atom_index: [target_atom_indices]}
        """
        neighbors = {}
        
        if query_atoms.n_atoms == 0 or target_atoms.n_atoms == 0:
            return neighbors
        
        # Compute distances using numpy
        query_pos = query_atoms.positions
        target_pos = target_atoms.positions
        distances = np.sqrt(((query_pos[:, np.newaxis, :] - target_pos[np.newaxis, :, :]) ** 2).sum(axis=2))
        
        # Find close contacts
        for i in range(distances.shape[0]):
            close_j = np.where(distances[i] < threshold)[0]
            if len(close_j) > 0:
                neighbors[query_atoms.indices[i]] = target_atoms.indices[close_j]
        
        return neighbors
    
    def detect_water_protein_dna_bridges(self) -> pd.DataFrame:
        """
        Detect water molecules bridging Protein and DNA.
        
        Critical for ETS-DNA recognition. Water molecules can mediate
        sequence-specific DNA recognition through coordinated hydration.
        
        Returns:
            pd.DataFrame: Columns [water_resid, protein_resid, dna_resid, 
                                   water_dists, bridge_type]
        """
        logger.info("Detecting Water-Protein-DNA bridges...")
        
        bridges = []
        
        if self.water.n_atoms == 0 or self.protein.n_atoms == 0 or self.dna.n_atoms == 0:
            logger.warning("Insufficient atoms for bridge detection")
            return pd.DataFrame()
        
        # Find water neighbors to protein and DNA
        water_protein_neighbors = self._find_neighbors(self.water, self.protein)
        water_dna_neighbors = self._find_neighbors(self.water, self.dna)
        
        # For each water, check if it connects protein and DNA
        for water_idx in self.water.indices:
            if water_idx not in water_protein_neighbors:
                continue
            if water_idx not in water_dna_neighbors:
                continue
            
            water_atom = self.u.atoms[water_idx]
            
            protein_neighbors = water_protein_neighbors[water_idx]
            dna_neighbors = water_dna_neighbors[water_idx]
            
            # Compute distances for this water to its neighbors
            water_pos = water_atom.position
            
            for protein_idx in protein_neighbors:
                protein_atom = self.u.atoms[protein_idx]
                water_protein_dist = np.linalg.norm(water_pos - protein_atom.position)
                
                for dna_idx in dna_neighbors:
                    dna_atom = self.u.atoms[dna_idx]
                    water_dna_dist = np.linalg.norm(water_pos - dna_atom.position)
                    
                    bridges.append({
                        'water_resid': water_atom.resnum,
                        'water_resname': water_atom.resname,
                        'protein_resid': protein_atom.resnum,
                        'protein_resname': protein_atom.resname,
                        'protein_atom': protein_atom.name,
                        'dna_resid': dna_atom.resnum,
                        'dna_resname': dna_atom.resname,
                        'dna_atom': dna_atom.name,
                        'water_protein_dist': water_protein_dist,
                        'water_dna_dist': water_dna_dist,
                        'bridge_type': 'Protein-Water-DNA'
                    })
        
        df = pd.DataFrame(bridges)
        self.bridge_summary['protein_dna'] = len(df)
        logger.info(f"Found {len(df)} Protein-Water-DNA bridges")
        
        return df
    
    def detect_water_ligand_protein_bridges(self) -> pd.DataFrame:
        """
        Detect water molecules bridging Ligand and Protein.
        
        Water bridges to the ligand-binding site can stabilize binding poses.
        
        Returns:
            pd.DataFrame: Water bridge details
        """
        if self.ligand is None or self.ligand.n_atoms == 0:
            logger.info("No ligand present, skipping Ligand-Water-Protein bridges")
            return pd.DataFrame()
        
        logger.info("Detecting Water-Ligand-Protein bridges...")
        
        bridges = []
        
        if self.water.n_atoms == 0 or self.protein.n_atoms == 0 or self.ligand.n_atoms == 0:
            return pd.DataFrame()
        
        water_ligand_neighbors = self._find_neighbors(self.water, self.ligand)
        water_protein_neighbors = self._find_neighbors(self.water, self.protein)
        
        for water_idx in self.water.indices:
            if water_idx not in water_ligand_neighbors:
                continue
            if water_idx not in water_protein_neighbors:
                continue
            
            water_atom = self.u.atoms[water_idx]
            water_pos = water_atom.position
            
            ligand_neighbors = water_ligand_neighbors[water_idx]
            protein_neighbors = water_protein_neighbors[water_idx]
            
            for ligand_idx in ligand_neighbors:
                ligand_atom = self.u.atoms[ligand_idx]
                water_ligand_dist = np.linalg.norm(water_pos - ligand_atom.position)
                
                for protein_idx in protein_neighbors:
                    protein_atom = self.u.atoms[protein_idx]
                    water_protein_dist = np.linalg.norm(water_pos - protein_atom.position)
                    
                    bridges.append({
                        'water_resid': water_atom.resnum,
                        'water_resname': water_atom.resname,
                        'ligand_atom': ligand_atom.name,
                        'ligand_resname': ligand_atom.resname,
                        'protein_resid': protein_atom.resnum,
                        'protein_resname': protein_atom.resname,
                        'protein_atom': protein_atom.name,
                        'water_ligand_dist': water_ligand_dist,
                        'water_protein_dist': water_protein_dist,
                        'bridge_type': 'Ligand-Water-Protein'
                    })
        
        df = pd.DataFrame(bridges)
        self.bridge_summary['ligand_protein'] = len(df)
        logger.info(f"Found {len(df)} Ligand-Water-Protein bridges")
        
        return df
    
    def detect_water_ligand_dna_bridges(self) -> pd.DataFrame:
        """
        Detect water molecules bridging Ligand and DNA.
        
        Critical for understanding how the drug interacts with DNA through
        the aqueous environment.
        
        Returns:
            pd.DataFrame: Water bridge details
        """
        if self.ligand is None or self.ligand.n_atoms == 0:
            logger.info("No ligand present, skipping Ligand-Water-DNA bridges")
            return pd.DataFrame()
        
        logger.info("Detecting Water-Ligand-DNA bridges...")
        
        bridges = []
        
        if self.water.n_atoms == 0 or self.ligand.n_atoms == 0 or self.dna.n_atoms == 0:
            return pd.DataFrame()
        
        water_ligand_neighbors = self._find_neighbors(self.water, self.ligand)
        water_dna_neighbors = self._find_neighbors(self.water, self.dna)
        
        for water_idx in self.water.indices:
            if water_idx not in water_ligand_neighbors:
                continue
            if water_idx not in water_dna_neighbors:
                continue
            
            water_atom = self.u.atoms[water_idx]
            water_pos = water_atom.position
            
            ligand_neighbors = water_ligand_neighbors[water_idx]
            dna_neighbors = water_dna_neighbors[water_idx]
            
            for ligand_idx in ligand_neighbors:
                ligand_atom = self.u.atoms[ligand_idx]
                water_ligand_dist = np.linalg.norm(water_pos - ligand_atom.position)
                
                for dna_idx in dna_neighbors:
                    dna_atom = self.u.atoms[dna_idx]
                    water_dna_dist = np.linalg.norm(water_pos - dna_atom.position)
                    
                    bridges.append({
                        'water_resid': water_atom.resnum,
                        'water_resname': water_atom.resname,
                        'ligand_atom': ligand_atom.name,
                        'ligand_resname': ligand_atom.resname,
                        'dna_resid': dna_atom.resnum,
                        'dna_resname': dna_atom.resname,
                        'dna_atom': dna_atom.name,
                        'water_ligand_dist': water_ligand_dist,
                        'water_dna_dist': water_dna_dist,
                        'bridge_type': 'Ligand-Water-DNA'
                    })
        
        df = pd.DataFrame(bridges)
        self.bridge_summary['ligand_dna'] = len(df)
        logger.info(f"Found {len(df)} Ligand-Water-DNA bridges")
        
        return df
    
    def detect_all_water_bridges(self) -> pd.DataFrame:
        """
        Detect all water bridges in the system.
        
        Returns:
            pd.DataFrame: Combined water bridge table
        """
        logger.info("=" * 60)
        logger.info("Detecting All Water Bridges (ETS Special)")
        logger.info("=" * 60)
        
        bridges_prot_dna = self.detect_water_protein_dna_bridges()
        bridges_lig_prot = self.detect_water_ligand_protein_bridges()
        bridges_lig_dna = self.detect_water_ligand_dna_bridges()
        
        # Combine all bridges
        all_bridges = pd.concat([bridges_prot_dna, bridges_lig_prot, bridges_lig_dna],
                                ignore_index=True, sort=False)
        
        self.water_bridges_df = all_bridges
        
        logger.info("-" * 60)
        logger.info(f"Total water bridges detected: {len(all_bridges)}")
        if len(all_bridges) > 0:
            logger.info("Water bridge types:")
            for btype, count in all_bridges['bridge_type'].value_counts().items():
                logger.info(f"  {btype}: {count}")
        logger.info("=" * 60)
        
        return all_bridges
    
    def get_water_bridge_interactions(self) -> pd.DataFrame:
        """
        Convert water bridges to interaction DataFrame format for integration
        with engine.py results.
        
        Returns:
            pd.DataFrame: Interactions with type='WaterBridge'
        """
        if self.water_bridges_df is None:
            self.detect_all_water_bridges()
        
        if len(self.water_bridges_df) == 0:
            return pd.DataFrame()
        
        # Reformat for compatibility with engine.py
        interactions = []
        for _, row in self.water_bridges_df.iterrows():
            interactions.append({
                'pair_type': row['bridge_type'],
                'interaction_type': 'WaterBridge',
                'distance': row.get('water_protein_dist', row.get('water_ligand_dist', 0)),
                'bridge_details': row.to_dict()
            })
        
        return pd.DataFrame(interactions)
    
    def save_water_bridges(self, output_path: str) -> str:
        """
        Save water bridge table to CSV.
        
        Args:
            output_path (str): Path to output CSV file
        
        Returns:
            str: Path to saved file
        """
        if self.water_bridges_df is None:
            self.detect_all_water_bridges()
        
        self.water_bridges_df.to_csv(output_path, index=False)
        logger.info(f"Saved water bridges to {output_path}")
        return output_path


def create_detector(universe: mda.Universe, atom_groups: Dict[str, mda.AtomGroup]) -> WaterBridgeDetector:
    """
    Convenience function to create a water bridge detector.
    
    Args:
        universe (mda.Universe): The loaded molecular system
        atom_groups (dict): Output from loader.get_atom_groups()
    
    Returns:
        WaterBridgeDetector: Initialized detector
    """
    return WaterBridgeDetector(
        universe,
        atom_groups['water'],
        atom_groups['protein'],
        atom_groups['dna'],
        atom_groups['ligand']
    )


if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python water_bridge.py <pdb_file>")
        sys.exit(1)
    
    from loader import load_pdb
    
    pdb_file = sys.argv[1]
    loader = load_pdb(pdb_file)
    
    detector = create_detector(loader.u, loader.get_atom_groups())
    bridges_df = detector.detect_all_water_bridges()
    
    print("\n=== Water Bridge Summary ===")
    print(bridges_df.groupby('bridge_type').size())
