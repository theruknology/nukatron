"""
Nukatron: Production-Grade Tri-Body Structural Analysis Pipeline
================================================================

A comprehensive Python 3.10+ framework for analyzing protein-DNA-drug
complexes with emphasis on ETS Transcription Factors and their
highly coordinated water-mediated interactions.

Modules:
  - loader: Robust PDB parsing and atom classification
  - engine: Tri-Body interaction fingerprinting
  - water_bridge: ETS-specific water bridge detection
  - visualizer: Interactive PyVis network visualization
  - batch_ml: Machine learning clustering of poses

Usage:
    from src.loader import load_pdb
    from src.engine import create_engine
    
    loader = load_pdb('protein.pdb')
    engine = create_engine(loader.u, loader.get_atom_groups())
    interactions = engine.compute_all_interactions()
"""

__version__ = '1.0.0'
__author__ = 'Nukatron Development Team'
__license__ = 'MIT'

from src.loader import PDBLoader, load_pdb
from src.engine import InteractionEngine, create_engine
from src.water_bridge import WaterBridgeDetector, create_detector
from src.visualizer import InteractionNetwork, create_visualizer
from src.batch_ml import BatchProcessor, InteractionFingerprint
from src.nucplot_legacy import NucplotExporter, create_legacy_exporter

__all__ = [
    'PDBLoader', 'load_pdb',
    'InteractionEngine', 'create_engine',
    'WaterBridgeDetector', 'create_detector',
    'InteractionNetwork', 'create_visualizer',
    'BatchProcessor', 'InteractionFingerprint',
    'NucplotExporter', 'create_legacy_exporter'
]
