#!/usr/bin/env python3
"""
Nukatron: Production-Grade Tri-Body Structural Analysis Pipeline
==================================================================
Main entry point with CLI interface for analyzing:
  - Single PDB files
  - Batch processing of docking poses
  - Interactive visualization
  - Machine learning clustering

Usage:
    python main.py --input <pdb_file> --output <output_dir> [--ligand <resname>]
    python main.py --batch <pdb_dir> --output <output_dir> --method dbscan
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Optional, List
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_output_directory(output_dir: str) -> Path:
    """
    Create output directory and subdirectories.
    
    Args:
        output_dir (str): Output directory path
    
    Returns:
        Path: Verified output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_path / 'interactions').mkdir(exist_ok=True)
    (output_path / 'visualizations').mkdir(exist_ok=True)
    (output_path / 'clustering').mkdir(exist_ok=True)
    
    logger.info(f"Output directory: {output_path}")
    return output_path


def analyze_single_pdb(pdb_file: str, output_dir: str, 
                      ligand_resnames: Optional[List[str]] = None,
                      legacy_ps: bool = False,
                      legacy_pdf: bool = False) -> dict:
    """
    Analyze a single PDB file (full pipeline).
    
    Args:
        pdb_file (str): Path to PDB file
        output_dir (str): Output directory
        ligand_resnames (list, optional): Custom ligand residue names
    
    Returns:
        dict: Analysis results and output file paths
    """
    from src.loader import load_pdb
    from src.engine import create_engine
    from src.water_bridge import create_detector
    from src.visualizer import create_visualizer
    
    output_path = setup_output_directory(output_dir)
    
    logger.info("=" * 70)
    logger.info(f"ANALYZING: {Path(pdb_file).name}")
    logger.info("=" * 70)
    
    results = {'input_file': pdb_file, 'outputs': {}}
    
    try:
        # Step 1: Load PDB
        logger.info("\n[1/5] Loading PDB file...")
        loader = load_pdb(pdb_file, ligand_resnames)
        atom_groups = loader.get_atom_groups()
        
        # Save residue summary
        residue_summary = loader.get_residue_summary()
        residue_csv = output_path / 'interactions' / 'residue_summary.csv'
        residue_summary.to_csv(residue_csv, index=False)
        results['outputs']['residue_summary'] = str(residue_csv)
        
        # Step 2: Compute interactions
        logger.info("\n[2/5] Computing Tri-Body interactions...")
        engine = create_engine(loader.u, atom_groups)
        interactions = engine.compute_all_interactions()
        
        # Save interactions
        interactions_csv = output_path / 'interactions' / 'interactions.csv'
        engine.save_interactions(str(interactions_csv))
        results['outputs']['interactions'] = str(interactions_csv)
        results['n_interactions'] = len(interactions)
        results['interaction_types'] = interactions['interaction_type'].value_counts().to_dict()
        
        # Step 3: Detect water bridges
        logger.info("\n[3/5] Detecting ETS-special water bridges...")
        detector = create_detector(loader.u, atom_groups)
        water_bridges = detector.detect_all_water_bridges()
        
        if len(water_bridges) > 0:
            water_bridges_csv = output_path / 'interactions' / 'water_bridges.csv'
            detector.save_water_bridges(str(water_bridges_csv))
            results['outputs']['water_bridges'] = str(water_bridges_csv)
            results['n_water_bridges'] = len(water_bridges)
            results['water_bridge_types'] = water_bridges['bridge_type'].value_counts().to_dict()
        else:
            logger.info("No water bridges detected")
            results['n_water_bridges'] = 0

        if legacy_ps or legacy_pdf:
            logger.info("\n[Legacy] Generating Nucplot-style PostScript schematic...")
            from src.nucplot_legacy import create_legacy_exporter

            exporter = create_legacy_exporter(
                interactions,
                water_bridges_df=water_bridges,
                residue_summary_df=residue_summary,
                ligand_label=(ligand_resnames[0] if ligand_resnames else None),
            )
            if legacy_ps:
                legacy_ps_path = output_path / 'output.ps'
                exporter.export(str(legacy_ps_path), title=Path(pdb_file).name)
                results['outputs']['legacy_ps'] = str(legacy_ps_path)
            if legacy_pdf:
                legacy_pdf_path = output_path / 'output.pdf'
                exporter.export(str(legacy_pdf_path), title=Path(pdb_file).name)
                results['outputs']['legacy_pdf'] = str(legacy_pdf_path)
        
        # Step 4: Interactive visualization
        logger.info("\n[4/5] Building interactive network visualization...")
        try:
            visualizer = create_visualizer(interactions, water_bridges)
            network_stats = visualizer.get_network_stats()
            
            network_html = output_path / 'visualizations' / 'network.html'
            visualizer.save_html(str(network_html))
            results['outputs']['network_visualization'] = str(network_html)
            results['network_stats'] = network_stats
            
            logger.info(f"Network: {network_stats['num_nodes']} nodes, "
                       f"{network_stats['num_edges']} edges")
        except ImportError:
            logger.warning("PyVis not available. Skipping visualization.")
        
        # Step 5: Summary report
        logger.info("\n[5/5] Generating summary report...")
        summary_json = output_path / 'analysis_summary.json'
        with open(summary_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        results['outputs']['summary'] = str(summary_json)
        
        logger.info("=" * 70)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 70)
        
        return results
    
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        raise


def batch_cluster_pdb_directory(pdb_directory: str, output_dir: str, 
                               method: str = 'dbscan', n_clusters: int = 5,
                               ligand_resnames: Optional[List[str]] = None) -> dict:
    """
    Batch process a directory of PDB files and cluster them.
    
    Args:
        pdb_directory (str): Directory containing PDB files
        output_dir (str): Output directory
        method (str): Clustering method ('dbscan' or 'kmeans')
        n_clusters (int): Number of clusters (for KMeans)
        ligand_resnames (list, optional): Custom ligand residue names
    
    Returns:
        dict: Clustering results and output paths
    """
    from src.batch_ml import BatchProcessor
    
    output_path = setup_output_directory(output_dir)
    
    logger.info("=" * 70)
    logger.info(f"BATCH PROCESSING: {pdb_directory}")
    logger.info("=" * 70)
    
    results = {'input_directory': pdb_directory, 'outputs': {}}
    
    try:
        # Process directory
        logger.info(f"\nProcessing PDBs from: {pdb_directory}")
        processor = BatchProcessor(pdb_directory, ligand_resnames)
        results_df = processor.process_directory()
        
        results['n_pdb_files'] = len(results_df)
        
        # Cluster
        logger.info(f"\nClustering with {method}...")
        results_clustered = processor.cluster_poses(results_df, method=method, n_clusters=n_clusters)
        
        # Save results
        clustering_csv = output_path / 'clustering' / 'clustering_results.csv'
        processor.save_results(results_clustered, str(clustering_csv))
        results['outputs']['clustering_results'] = str(clustering_csv)
        
        # Clustering statistics
        results['n_clusters'] = results_clustered['cluster_id'].nunique()
        results['cluster_distribution'] = results_clustered['cluster_id'].value_counts().to_dict()
        results['avg_clash_score'] = float(results_clustered['clash_score'].mean())
        results['avg_binding_score'] = float(results_clustered['binding_score'].mean())
        
        # Save summary
        summary_json = output_path / 'clustering_summary.json'
        with open(summary_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        results['outputs']['summary'] = str(summary_json)
        
        logger.info("=" * 70)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("=" * 70)
        
        return results
    
    except Exception as e:
        logger.error(f"Error during batch processing: {e}", exc_info=True)
        raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Nukatron: Tri-Body Structural Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single PDB analysis
  python main.py --input myprotein.pdb --output results/

  # With custom ligand
  python main.py --input myprotein.pdb --output results/ --ligand LIG

  # Batch clustering
  python main.py --batch pdbs/ --output results/ --method dbscan

  # Batch with KMeans
  python main.py --batch pdbs/ --output results/ --method kmeans --n_clusters 10
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Path to single PDB file')
    input_group.add_argument('--batch', type=str, help='Path to directory of PDB files')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output directory')
    
    # Ligand specification
    parser.add_argument('--ligand', '-l', type=str, action='append', dest='ligand_resnames',
                       help='Ligand residue name(s) (can be specified multiple times)')
    
    # Clustering options
    parser.add_argument('--method', type=str, choices=['dbscan', 'kmeans'], 
                       default='dbscan',
                       help='Clustering method (default: dbscan)')
    parser.add_argument('--n_clusters', type=int, default=5,
                       help='Number of clusters for KMeans (default: 5)')
    
    # Verbosity
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose logging')

    parser.add_argument('--legacy-ps', action='store_true',
                       help='Generate Nucplot-style legacy PostScript schematic as output.ps')

    parser.add_argument('--legacy-pdf', action='store_true',
                       help='Generate Nucplot-style legacy PDF schematic as output.pdf')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display header
    logger.info("=" * 70)
    logger.info("NUKATRON: Production-Grade Tri-Body Structural Analysis")
    logger.info("=" * 70)
    logger.info(f"Start time: {pd.Timestamp.now()}")
    
    try:
        if args.input:
            # Single file analysis
            if not Path(args.input).exists():
                logger.error(f"PDB file not found: {args.input}")
                sys.exit(1)
            
            results = analyze_single_pdb(
                args.input,
                args.output,
                args.ligand_resnames,
                legacy_ps=args.legacy_ps,
                legacy_pdf=args.legacy_pdf,
            )
            
            # Print results
            logger.info("\n" + "=" * 70)
            logger.info("ANALYSIS RESULTS")
            logger.info("=" * 70)
            logger.info(f"Interactions found: {results.get('n_interactions', 0)}")
            logger.info(f"Water bridges found: {results.get('n_water_bridges', 0)}")
            if 'network_stats' in results:
                logger.info(f"Network nodes: {results['network_stats'].get('num_nodes', 0)}")
                logger.info(f"Network edges: {results['network_stats'].get('num_edges', 0)}")
            
            logger.info("\nOutput files:")
            for key, value in results['outputs'].items():
                logger.info(f"  {key}: {value}")
        
        elif args.batch:
            # Batch processing
            if not Path(args.batch).is_dir():
                logger.error(f"Directory not found: {args.batch}")
                sys.exit(1)
            
            results = batch_cluster_pdb_directory(
                args.batch, args.output,
                method=args.method,
                n_clusters=args.n_clusters,
                ligand_resnames=args.ligand_resnames
            )
            
            # Print results
            logger.info("\n" + "=" * 70)
            logger.info("BATCH CLUSTERING RESULTS")
            logger.info("=" * 70)
            logger.info(f"PDB files processed: {results.get('n_pdb_files', 0)}")
            logger.info(f"Number of clusters: {results.get('n_clusters', 0)}")
            logger.info(f"Average clash score: {results.get('avg_clash_score', 0):.3f}")
            logger.info(f"Average binding score: {results.get('avg_binding_score', 0):.3f}")
            
            logger.info("\nCluster distribution:")
            for cluster_id, count in sorted(results.get('cluster_distribution', {}).items()):
                logger.info(f"  Cluster {cluster_id}: {count} poses")
            
            logger.info("\nOutput files:")
            for key, value in results['outputs'].items():
                logger.info(f"  {key}: {value}")
        
        logger.info("\n" + "=" * 70)
        logger.info("Nukatron completed successfully!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    import pandas as pd
    main()
