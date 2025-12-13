"""
Nukatron Visualizer Module: Interactive Network Visualization
==============================================================
Uses PyVis to generate interactive HTML network visualizations.

Layout:
  - Center (Red): Drug/Ligand
  - Inner Orbit (Orange): DNA bases
  - Outer Orbit (Blue): Protein residues
  - Small Light Blue: Water molecules
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import networkx as nx

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    logging.warning("PyVis not available. HTML visualization will be skipped.")

logger = logging.getLogger(__name__)


class InteractionNetwork:
    """
    Interactive network visualization of molecular interactions.
    """
    
    def __init__(self, interactions_df: pd.DataFrame, water_bridges_df: Optional[pd.DataFrame] = None):
        """
        Initialize the interaction network.
        
        Args:
            interactions_df (pd.DataFrame): Output from engine.compute_all_interactions()
            water_bridges_df (pd.DataFrame, optional): Output from water_bridge detector
        """
        self.interactions = interactions_df
        self.water_bridges = water_bridges_df if water_bridges_df is not None else pd.DataFrame()
        
        self.graph = None
        self.pyvis_net = None
    
    def _build_networkx_graph(self) -> nx.Graph:
        """
        Build a NetworkX graph from interactions.
        
        Returns:
            nx.Graph: Interaction network
        """
        G = nx.Graph()
        
        # Add nodes and edges from interactions
        for _, row in self.interactions.iterrows():
            pair_type = row['pair_type']
            
            if pair_type == 'Ligand-Protein':
                ligand_node = f"LIG_{row['ligand_resid']}"
                protein_node = f"PRO_{row['protein_resname']}_{row['protein_resid']}"
                
                G.add_node(ligand_node, node_type='ligand', resid=row['ligand_resid'])
                G.add_node(protein_node, node_type='protein', 
                          resname=row['protein_resname'], resid=row['protein_resid'])
                
                G.add_edge(ligand_node, protein_node, 
                          interaction=row['interaction_type'],
                          distance=row['distance'])
            
            elif pair_type == 'Ligand-DNA':
                ligand_node = f"LIG_{row['ligand_resid']}"
                dna_node = f"DNA_{row['dna_resname']}_{row['dna_resid']}"
                
                G.add_node(ligand_node, node_type='ligand', resid=row['ligand_resid'])
                G.add_node(dna_node, node_type='dna',
                          resname=row['dna_resname'], resid=row['dna_resid'])
                
                G.add_edge(ligand_node, dna_node,
                          interaction=row['interaction_type'],
                          distance=row['distance'])
            
            elif pair_type == 'Protein-DNA':
                protein_node = f"PRO_{row['protein_resname']}_{row['protein_resid']}"
                dna_node = f"DNA_{row['dna_resname']}_{row['dna_resid']}"
                
                G.add_node(protein_node, node_type='protein',
                          resname=row['protein_resname'], resid=row['protein_resid'])
                G.add_node(dna_node, node_type='dna',
                          resname=row['dna_resname'], resid=row['dna_resid'])
                
                G.add_edge(protein_node, dna_node,
                          interaction=row['interaction_type'],
                          distance=row['distance'])
        
        # Add water bridge nodes and edges
        if len(self.water_bridges) > 0:
            for _, row in self.water_bridges.iterrows():
                water_node = f"WAT_{row['water_resid']}"
                bridge_type = row['bridge_type']
                
                G.add_node(water_node, node_type='water', resid=row['water_resid'])
                
                if bridge_type == 'Protein-Water-DNA':
                    protein_node = f"PRO_{row['protein_resname']}_{row['protein_resid']}"
                    dna_node = f"DNA_{row['dna_resname']}_{row['dna_resid']}"
                    
                    G.add_node(protein_node, node_type='protein',
                              resname=row['protein_resname'], resid=row['protein_resid'])
                    G.add_node(dna_node, node_type='dna',
                              resname=row['dna_resname'], resid=row['dna_resid'])
                    
                    G.add_edge(protein_node, water_node, interaction='WaterBridge')
                    G.add_edge(water_node, dna_node, interaction='WaterBridge')
                
                elif bridge_type == 'Ligand-Water-Protein':
                    ligand_node = f"LIG_{row['ligand_resname']}"
                    protein_node = f"PRO_{row['protein_resname']}_{row['protein_resid']}"
                    
                    G.add_node(ligand_node, node_type='ligand')
                    G.add_node(protein_node, node_type='protein',
                              resname=row['protein_resname'], resid=row['protein_resid'])
                    
                    G.add_edge(ligand_node, water_node, interaction='WaterBridge')
                    G.add_edge(water_node, protein_node, interaction='WaterBridge')
                
                elif bridge_type == 'Ligand-Water-DNA':
                    ligand_node = f"LIG_{row['ligand_resname']}"
                    dna_node = f"DNA_{row['dna_resname']}_{row['dna_resid']}"
                    
                    G.add_node(ligand_node, node_type='ligand')
                    G.add_node(dna_node, node_type='dna',
                              resname=row['dna_resname'], resid=row['dna_resid'])
                    
                    G.add_edge(ligand_node, water_node, interaction='WaterBridge')
                    G.add_edge(water_node, dna_node, interaction='WaterBridge')
        
        self.graph = G
        logger.info(f"Built network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def _get_node_color(self, node: str, node_data: Dict) -> str:
        """Get node color based on type."""
        node_type = node_data.get('node_type', 'unknown')
        
        color_map = {
            'ligand': '#FF0000',      # Red
            'dna': '#FFA500',         # Orange
            'protein': '#0000FF',     # Blue
            'water': '#ADD8E6'        # Light Blue
        }
        
        return color_map.get(node_type, '#808080')  # Gray for unknown
    
    def _get_node_size(self, node: str, node_data: Dict) -> int:
        """Get node size based on type."""
        node_type = node_data.get('node_type', 'unknown')
        
        size_map = {
            'ligand': 40,
            'dna': 30,
            'protein': 25,
            'water': 15
        }
        
        return size_map.get(node_type, 20)
    
    def _get_edge_color(self, edge_data: Dict) -> str:
        """Get edge color based on interaction type."""
        interaction = edge_data.get('interaction', 'Contact')
        
        color_map = {
            'HBond': '#00AA00',       # Green
            'VdW': '#CCCCCC',         # Light gray
            'PiStacking': '#FFFF00',  # Yellow
            'WaterBridge': '#87CEEB'  # Sky blue
        }
        
        return color_map.get(interaction, '#CCCCCC')
    
    def _get_edge_width(self, edge_data: Dict) -> float:
        """Get edge width based on interaction type."""
        interaction = edge_data.get('interaction', 'Contact')
        distance = edge_data.get('distance', 0)
        
        # Width based on interaction strength (shorter = stronger)
        if interaction == 'HBond':
            return 3.0
        elif interaction == 'PiStacking':
            return 2.5
        elif interaction == 'WaterBridge':
            return 2.0
        else:
            return 1.0
    
    def build_pyvis_network(self, physics_enabled: bool = True) -> 'Network':
        """
        Build an interactive PyVis network.
        
        Args:
            physics_enabled (bool): Enable physics simulation for layout
        
        Returns:
            pyvis.network.Network: Configured network object
        """
        if not PYVIS_AVAILABLE:
            raise ImportError("PyVis is not installed. Install with: pip install pyvis")
        
        if self.graph is None:
            self._build_networkx_graph()
        
        # Create PyVis network
        net = Network(height='750px', width='100%', directed=False)
        net.from_nx(self.graph)
        
        # Configure physics
        net.toggle_physics(physics_enabled)
        
        # Update node colors and sizes
        for node in net.nodes:
            node_id = node['id']
            node_data = self.graph.nodes[node_id]
            
            node['color'] = self._get_node_color(node_id, node_data)
            node['size'] = self._get_node_size(node_id, node_data)
            node['title'] = node_id  # Hover tooltip
        
        # Update edge colors and widths
        for edge in net.edges:
            source = edge['from']
            target = edge['to']
            
            if self.graph.has_edge(source, target):
                edge_data = self.graph[source][target]
                edge['color'] = self._get_edge_color(edge_data)
                edge['width'] = self._get_edge_width(edge_data)
                
                # Add interaction type to hover
                interaction = edge_data.get('interaction', 'Unknown')
                distance = edge_data.get('distance', 0)
                edge['title'] = f"{interaction} ({distance:.2f} Å)"
        
        # Configure layout options
        net.toggle_physics(physics_enabled)
        
        self.pyvis_net = net
        logger.info("PyVis network constructed")
        
        return net
    
    def save_html(self, output_path: str, show_physics: bool = True) -> str:
        """
        Save interactive network visualization to HTML.
        
        Args:
            output_path (str): Path to output HTML file
            show_physics (bool): Show physics control buttons
        
        Returns:
            str: Path to saved HTML file
        """
        if self.pyvis_net is None:
            self.build_pyvis_network(physics_enabled=show_physics)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.pyvis_net.write_html(str(output_path))
        logger.info(f"Saved interactive network to {output_path}")
        
        return str(output_path)
    
    def get_network_stats(self) -> Dict:
        """
        Get summary statistics of the network.
        
        Returns:
            dict: Network statistics
        """
        if self.graph is None:
            self._build_networkx_graph()
        
        # Count node types
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            ntype = data.get('node_type', 'unknown')
            node_types[ntype] = node_types.get(ntype, 0) + 1
        
        # Count edge types
        edge_types = {}
        for source, target, data in self.graph.edges(data=True):
            etype = data.get('interaction', 'unknown')
            edge_types[etype] = edge_types.get(etype, 0) + 1
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'node_types': node_types,
            'edge_types': edge_types,
            'density': nx.density(self.graph)
        }


def create_visualizer(interactions_df: pd.DataFrame, 
                     water_bridges_df: Optional[pd.DataFrame] = None) -> InteractionNetwork:
    """
    Convenience function to create an interaction network visualizer.
    
    Args:
        interactions_df (pd.DataFrame): Output from engine.compute_all_interactions()
        water_bridges_df (pd.DataFrame, optional): Output from water_bridge detector
    
    Returns:
        InteractionNetwork: Initialized visualizer
    """
    return InteractionNetwork(interactions_df, water_bridges_df)


if __name__ == "__main__":
    # Example usage
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <pdb_file> [output.html]")
        sys.exit(1)
    
    from loader import load_pdb
    from engine import create_engine
    from water_bridge import create_detector
    
    pdb_file = sys.argv[1]
    output_html = sys.argv[2] if len(sys.argv) > 2 else "network.html"
    
    loader = load_pdb(pdb_file)
    atom_groups = loader.get_atom_groups()
    
    engine = create_engine(loader.u, atom_groups)
    interactions = engine.compute_all_interactions()
    
    detector = create_detector(loader.u, atom_groups)
    water_bridges = detector.detect_all_water_bridges()
    
    visualizer = create_visualizer(interactions, water_bridges)
    stats = visualizer.get_network_stats()
    print("\n=== Network Statistics ===")
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Node Types: {stats['node_types']}")
    print(f"Edge Types: {stats['edge_types']}")
    
    if PYVIS_AVAILABLE:
        visualizer.save_html(output_html)
        print(f"\nSaved visualization to {output_html}")
    else:
        print("PyVis not available. Skipping HTML visualization.")
