"""
Nukatron Batch ML Module: AI-Driven Clustering Pipeline
=========================================================
Process a directory of docking poses and cluster them based on:
  1. Interaction fingerprints (bit vectors)
  2. DBSCAN or KMeans clustering
  3. Clash score calculation

Suitable for analyzing ~1000 docking poses in parallel.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

logger = logging.getLogger(__name__)


class InteractionFingerprint:
    """
    Generate bit vector fingerprints from interactions.
    
    Each bit represents whether a specific residue (protein or DNA) 
    interacts with the ligand (1) or not (0).
    """
    
    def __init__(self):
        """Initialize fingerprint generator."""
        self.residue_vocab = set()
        self.residue_list = []
    
    def build_vocabulary(self, interactions_list: List[pd.DataFrame]) -> None:
        """
        Build vocabulary of all residues across all interactions.
        
        Args:
            interactions_list (list): List of interaction DataFrames
        """
        for interactions in interactions_list:
            if len(interactions) == 0:
                continue
            
            # Extract protein residues
            if 'protein_resid' in interactions.columns:
                self.residue_vocab.update(interactions['protein_resid'].unique())
            
            # Extract DNA residues
            if 'dna_resid' in interactions.columns:
                self.residue_vocab.update(interactions['dna_resid'].unique())
        
        self.residue_list = sorted(list(self.residue_vocab))
        logger.info(f"Built vocabulary with {len(self.residue_list)} unique residues")
    
    def generate_fingerprint(self, interactions: pd.DataFrame) -> np.ndarray:
        """
        Generate bit vector fingerprint for a single pose.
        
        Args:
            interactions (pd.DataFrame): Interaction DataFrame for one pose
        
        Returns:
            np.ndarray: Binary bit vector of shape (n_residues,)
        """
        fingerprint = np.zeros(len(self.residue_list))
        
        if len(interactions) == 0:
            return fingerprint
        
        # Mark residues that interact with ligand
        interacting_residues = set()
        
        if 'protein_resid' in interactions.columns:
            # Ligand-Protein interactions
            lig_prot = interactions[interactions['pair_type'] == 'Ligand-Protein']
            interacting_residues.update(lig_prot['protein_resid'].unique())
        
        if 'dna_resid' in interactions.columns:
            # Ligand-DNA interactions
            lig_dna = interactions[interactions['pair_type'] == 'Ligand-DNA']
            interacting_residues.update(lig_dna['dna_resid'].unique())
        
        # Set bits
        for residue in interacting_residues:
            if residue in self.residue_vocab:
                idx = self.residue_list.index(residue)
                fingerprint[idx] = 1
        
        return fingerprint
    
    def get_vocabulary_size(self) -> int:
        """Get size of residue vocabulary."""
        return len(self.residue_list)


class ClashScoreCalculator:
    """Calculate clash scores for poses."""
    
    @staticmethod
    def calculate_clash_score(interactions: pd.DataFrame) -> float:
        """
        Calculate a simple clash score based on Van der Waals violations.
        
        Higher score = more clashes/worse pose
        
        Args:
            interactions (pd.DataFrame): Interaction DataFrame
        
        Returns:
            float: Clash score (0-1)
        """
        if len(interactions) == 0:
            return 0.0
        
        # Count Van der Waals contacts (close, unfavorable)
        vdw_contacts = interactions[interactions['interaction_type'] == 'VdW']
        
        if len(vdw_contacts) == 0:
            return 0.0
        
        # VdW violations: distance < 3.0 Å (typical VdW sum ~3.4)
        violations = vdw_contacts[vdw_contacts['distance'] < 3.0]
        
        clash_score = len(violations) / len(interactions)
        return min(clash_score, 1.0)  # Normalize to [0, 1]
    
    @staticmethod
    def calculate_binding_score(interactions: pd.DataFrame) -> float:
        """
        Calculate a binding quality score based on favorable interactions.
        
        Higher score = better binding (more HBonds, better geometry)
        
        Args:
            interactions (pd.DataFrame): Interaction DataFrame
        
        Returns:
            float: Binding score (0-1)
        """
        if len(interactions) == 0:
            return 0.0
        
        # Count favorable interactions
        hbonds = len(interactions[interactions['interaction_type'] == 'HBond'])
        pi_stacking = len(interactions[interactions['interaction_type'] == 'PiStacking'])
        
        # Weighted score
        favorable = hbonds * 2 + pi_stacking * 1.5
        total = len(interactions)
        
        if total == 0:
            return 0.0
        
        binding_score = favorable / (total * 2)  # Normalize
        return min(binding_score, 1.0)


class PDBClusterer:
    """
    Cluster PDB poses based on interaction fingerprints.
    """
    
    def __init__(self, fingerprinter: InteractionFingerprint):
        """
        Initialize the clusterer.
        
        Args:
            fingerprinter (InteractionFingerprint): Fingerprint generator
        """
        self.fingerprinter = fingerprinter
        self.features = None
        self.feature_scaler = StandardScaler()
        self.cluster_labels = None
        self.n_clusters = 0
    
    def fit(self, features: np.ndarray, method: str = 'dbscan', n_clusters: int = 5) -> np.ndarray:
        """
        Fit clustering model to features.
        
        Args:
            features (np.ndarray): (n_samples, n_features) bit vectors
            method (str): 'dbscan' or 'kmeans'
            n_clusters (int): Number of clusters (for KMeans)
        
        Returns:
            np.ndarray: Cluster labels
        """
        logger.info(f"Clustering {len(features)} samples with {method}")
        
        # Standardize features
        features_scaled = self.feature_scaler.fit_transform(features)
        self.features = features_scaled
        
        if method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            self.cluster_labels = clusterer.fit_predict(features_scaled)
            self.n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
            
            logger.info(f"DBSCAN found {self.n_clusters} clusters")
            logger.info(f"Noise points: {sum(self.cluster_labels == -1)}")
        
        elif method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = clusterer.fit_predict(features_scaled)
            self.n_clusters = n_clusters
            
            inertia = clusterer.inertia_
            logger.info(f"KMeans converged with inertia: {inertia:.3f}")
        
        # Calculate metrics
        self._calculate_metrics(features_scaled)
        
        return self.cluster_labels
    
    def _calculate_metrics(self, features: np.ndarray) -> None:
        """Calculate clustering quality metrics."""
        if len(set(self.cluster_labels)) < 2:
            logger.warning("Cannot calculate metrics with < 2 clusters")
            return
        
        # Silhouette score (higher is better, range: -1 to 1)
        sil_score = silhouette_score(features, self.cluster_labels)
        logger.info(f"Silhouette Score: {sil_score:.3f}")
        
        # Davies-Bouldin Index (lower is better)
        if len(set(self.cluster_labels)) > 1:
            db_index = davies_bouldin_score(features, self.cluster_labels)
            logger.info(f"Davies-Bouldin Index: {db_index:.3f}")


class BatchProcessor:
    """
    Process a directory of PDB files and cluster them.
    """
    
    def __init__(self, pdb_directory: str, ligand_resnames: Optional[List[str]] = None):
        """
        Initialize batch processor.
        
        Args:
            pdb_directory (str): Directory containing PDB files
            ligand_resnames (list, optional): Custom ligand residue names
        """
        self.pdb_dir = Path(pdb_directory)
        self.ligand_resnames = ligand_resnames
        
        self.results = []
        self.fingerprinter = InteractionFingerprint()
        self.clusterer = None
    
    def process_directory(self) -> pd.DataFrame:
        """
        Process all PDB files in directory.
        
        Returns:
            pd.DataFrame: Results with filename, fingerprint, interactions
        """
        from loader import load_pdb
        from engine import create_engine
        
        pdb_files = sorted(self.pdb_dir.glob('*.pdb'))
        
        if not pdb_files:
            logger.warning(f"No PDB files found in {self.pdb_dir}")
            return pd.DataFrame()
        
        logger.info(f"Found {len(pdb_files)} PDB files")
        
        interactions_list = []
        
        # First pass: collect all interactions and build vocabulary
        logger.info("=" * 60)
        logger.info("PASS 1: Building interaction vocabulary")
        logger.info("=" * 60)
        
        for pdb_file in tqdm(pdb_files, desc="Building vocabulary"):
            try:
                loader = load_pdb(str(pdb_file), self.ligand_resnames)
                engine = create_engine(loader.u, loader.get_atom_groups())
                interactions = engine.compute_all_interactions()
                interactions_list.append(interactions)
            except Exception as e:
                logger.error(f"Failed to process {pdb_file.name}: {e}")
                interactions_list.append(pd.DataFrame())
        
        # Build vocabulary
        self.fingerprinter.build_vocabulary(interactions_list)
        
        # Second pass: generate fingerprints and scores
        logger.info("=" * 60)
        logger.info("PASS 2: Generating fingerprints and scores")
        logger.info("=" * 60)
        
        fingerprints = []
        for pdb_file, interactions in tqdm(zip(pdb_files, interactions_list),
                                          total=len(pdb_files),
                                          desc="Generating fingerprints"):
            fingerprint = self.fingerprinter.generate_fingerprint(interactions)
            fingerprints.append(fingerprint)
            
            clash_score = ClashScoreCalculator.calculate_clash_score(interactions)
            binding_score = ClashScoreCalculator.calculate_binding_score(interactions)
            
            self.results.append({
                'filename': pdb_file.name,
                'n_interactions': len(interactions),
                'clash_score': clash_score,
                'binding_score': binding_score
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Add fingerprints as separate columns
        fp_array = np.array(fingerprints)
        for i, residue in enumerate(self.fingerprinter.residue_list):
            results_df[f'res_{residue}'] = fp_array[:, i]
        
        logger.info(f"Processed {len(pdb_files)} PDB files")
        logger.info(f"Generated {fp_array.shape[1]} features per fingerprint")
        
        return results_df
    
    def cluster_poses(self, results_df: pd.DataFrame, method: str = 'dbscan',
                     n_clusters: int = 5) -> pd.DataFrame:
        """
        Cluster poses based on fingerprints.
        
        Args:
            results_df (pd.DataFrame): Output from process_directory()
            method (str): 'dbscan' or 'kmeans'
            n_clusters (int): Number of clusters (for KMeans)
        
        Returns:
            pd.DataFrame: Results with added 'cluster_id' column
        """
        logger.info("=" * 60)
        logger.info(f"Clustering with method: {method}")
        logger.info("=" * 60)
        
        # Extract fingerprint features
        fp_columns = [col for col in results_df.columns if col.startswith('res_')]
        features = results_df[fp_columns].values
        
        # Cluster
        self.clusterer = PDBClusterer(self.fingerprinter)
        cluster_labels = self.clusterer.fit(features, method=method, n_clusters=n_clusters)
        
        results_df['cluster_id'] = cluster_labels
        
        logger.info("-" * 60)
        logger.info("Cluster distribution:")
        print(results_df['cluster_id'].value_counts().sort_index())
        logger.info("=" * 60)
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, output_csv: str) -> str:
        """
        Save clustering results to CSV.
        
        Args:
            results_df (pd.DataFrame): Clustering results
            output_csv (str): Output CSV path
        
        Returns:
            str: Path to saved file
        """
        # Save simplified summary (without fingerprint columns)
        summary_cols = ['filename', 'n_interactions', 'clash_score', 'binding_score', 'cluster_id']
        available_cols = [col for col in summary_cols if col in results_df.columns]
        summary_df = results_df[available_cols]
        
        summary_df.to_csv(output_csv, index=False)
        logger.info(f"Saved results to {output_csv}")
        
        return output_csv


def process_batch(pdb_directory: str, output_csv: str, 
                 method: str = 'dbscan', n_clusters: int = 5,
                 ligand_resnames: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convenience function to process batch of PDBs and cluster them.
    
    Args:
        pdb_directory (str): Directory with PDB files
        output_csv (str): Output CSV path
        method (str): 'dbscan' or 'kmeans'
        n_clusters (int): Number of clusters (for KMeans)
        ligand_resnames (list, optional): Custom ligand residue names
    
    Returns:
        pd.DataFrame: Clustering results
    """
    processor = BatchProcessor(pdb_directory, ligand_resnames)
    results = processor.process_directory()
    results_clustered = processor.cluster_poses(results, method=method, n_clusters=n_clusters)
    processor.save_results(results_clustered, output_csv)
    
    return results_clustered


if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 3:
        print("Usage: python batch_ml.py <pdb_directory> <output.csv> [method] [n_clusters]")
        print("  method: 'dbscan' (default) or 'kmeans'")
        print("  n_clusters: number of clusters for KMeans (default: 5)")
        sys.exit(1)
    
    pdb_dir = sys.argv[1]
    output_csv = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else 'dbscan'
    n_clusters = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    
    results = process_batch(pdb_dir, output_csv, method=method, n_clusters=n_clusters)
    print("\n=== Clustering Summary ===")
    print(results[['filename', 'cluster_id', 'clash_score', 'binding_score']].head(10))
