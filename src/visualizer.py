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
            'ligand': '#FF5A67',      # Soft red
            'dna': '#F4B86A',         # Soft orange
            'protein': '#5AA2FF',     # Soft blue
            'water': '#7EC8E3'        # Soft light blue
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
            'HBond': '#4BD37C',       # Soft green
            'VdW': '#6B7280',         # Neutral gray
            'PiStacking': '#F2D06B',  # Soft yellow
            'WaterBridge': '#66C7FF'  # Soft sky blue
        }
        
        return color_map.get(interaction, '#6B7280')
    
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

        # Dark theme + readable labels + physics (avoid net.toggle_physics: some PyVis
        # versions store options as dict after set_options and then toggle_physics breaks)
        net.set_options(
            '{'
            f'  "physics": {{"enabled": {"true" if physics_enabled else "false"}}},'
            '  "nodes": {"font": {"color": "#E6EDF3", "size": 14, "face": "system-ui"}},'
            '  "edges": {"font": {"color": "#C7D0DA", "size": 12, "face": "system-ui"}},'
            '  "interaction": {"hover": true, "tooltipDelay": 120}'
            '}'
        )
        
        # Update node colors and sizes
        for node in net.nodes:
            node_id = node['id']
            node_data = self.graph.nodes[node_id]
            
            node['color'] = self._get_node_color(node_id, node_data)
            node['size'] = self._get_node_size(node_id, node_data)
            node['title'] = node_id  # Hover tooltip
            node['font'] = {
                'color': '#E6EDF3',
                'strokeWidth': 3,
                'strokeColor': '#0B0F14'
            }
        
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
        
        self.pyvis_net = net
        logger.info("PyVis network constructed")
        
        return net

    def _build_legend_html(self) -> str:
        node_legend = [
            ("Drug/Ligand", "#FF5A67"),
            ("DNA bases", "#F4B86A"),
            ("Protein residues", "#5AA2FF"),
            ("Water molecules", "#7EC8E3"),
        ]

        edge_legend = [
            ("Hydrogen bond (HBond)", "#4BD37C"),
            ("Van der Waals (VdW)", "#6B7280"),
            ("Pi stacking (PiStacking)", "#F2D06B"),
            ("Water bridge (WaterBridge)", "#66C7FF"),
        ]

        def swatch(label: str, color: str) -> str:
            return (
                '<div class="nuk-legend-item">'
                f'<span class="nuk-swatch" style="background:{color}"></span>'
                f'<span class="nuk-legend-label">{label}</span>'
                '</div>'
            )

        node_html = "".join([swatch(label, color) for label, color in node_legend])
        edge_html = "".join([swatch(label, color) for label, color in edge_legend])

        stats = None
        try:
            stats = self.get_network_stats()
        except Exception:
            stats = None

        stats_html = ""
        if isinstance(stats, dict) and 'num_nodes' in stats and 'num_edges' in stats:
            stats_html = (
                '<div class="nuk-stats">'
                f"<span><strong>Nodes</strong>: {stats['num_nodes']}</span>"
                f"<span><strong>Edges</strong>: {stats['num_edges']}</span>"
                '</div>'
            )

        return (
            '<div id="nuk-infobar">'
            '  <div class="nuk-info-left">'
            '    <div class="nuk-section">'
            '      <div class="nuk-section-title">Nodes</div>'
            f'      <div class="nuk-legend-grid">{node_html}</div>'
            '    </div>'
            '    <div class="nuk-section">'
            '      <div class="nuk-section-title">Edges</div>'
            f'      <div class="nuk-legend-grid">{edge_html}</div>'
            '    </div>'
            '  </div>'
            '  <div class="nuk-info-right">'
            f'    {stats_html}'
            '    <div class="nuk-help">'
            '      <div><strong>Controls</strong>: drag to move, scroll to zoom, click node/edge for details.</div>'
            '    </div>'
            '  </div>'
            '</div>'
        )

    def _inject_ui_into_html(self, html: str, initial_physics_enabled: bool = True) -> str:
        if 'id="nuk-toolbar"' in html or "id='nuk-toolbar'" in html:
            return html

        ui_html = (
            '<div id="nuk-ui">'
            '  <div id="nuk-toolbar">'
            '    <div class="nuk-title">Nukatron Network</div>'
            '    <div class="nuk-actions">'
            '      <button type="button" id="nuk-fit" class="nuk-btn">Fit</button>'
            '      <button type="button" id="nuk-center" class="nuk-btn">Center</button>'
            '      <button type="button" id="nuk-physics" class="nuk-btn">Toggle physics</button>'
            '      <button type="button" id="nuk-toggle-info" class="nuk-btn">Legend</button>'
            '      <span id="nuk-physics-state" class="nuk-pill">Physics: on</span>'
            '    </div>'
            '  </div>'
            f'{self._build_legend_html()}'
            '</div>'
        )

        css = (
            '<style id="nuk-ui-style">'
            '  :root{--nuk-toolbar-h:44px;--nuk-infobar-h:110px;--nuk-bg:#0b0f14;--nuk-bg2:#0f1620;--nuk-fg:#e6edf3;--nuk-muted:#9aa4af;--nuk-border:#233041;--nuk-accent:#4ea1ff;}'
            '  html,body{height:100%;}'
            '  body{margin:0;background:var(--nuk-bg);}'
            '  #nuk-ui{position:fixed;left:0;right:0;top:0;z-index:9999;font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--nuk-fg);}'
            '  #nuk-toolbar{height:var(--nuk-toolbar-h);display:flex;align-items:center;justify-content:space-between;gap:12px;padding:0 12px;background:linear-gradient(180deg,var(--nuk-bg),var(--nuk-bg2));border-bottom:1px solid var(--nuk-border);}'
            '  .nuk-title{font-weight:600;font-size:13px;letter-spacing:.2px;}'
            '  .nuk-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end;}'
            '  .nuk-btn{background:#111a26;border:1px solid var(--nuk-border);color:var(--nuk-fg);padding:6px 10px;border-radius:8px;font-size:12px;cursor:pointer;}'
            '  .nuk-btn:hover{border-color:#2f4157;}'
            '  .nuk-pill{display:inline-flex;align-items:center;padding:5px 10px;border-radius:999px;background:#0f1d2f;border:1px solid var(--nuk-border);font-size:12px;color:var(--nuk-muted);}'
            '  #nuk-infobar{display:flex;gap:16px;align-items:flex-start;justify-content:space-between;padding:10px 12px;background:rgba(11,15,20,.92);backdrop-filter:blur(6px);border-bottom:1px solid var(--nuk-border);}'
            '  body.nuk-hide-info #nuk-infobar{display:none;}'
            '  .nuk-info-left{display:flex;gap:16px;flex-wrap:wrap;}'
            '  .nuk-info-right{display:flex;flex-direction:column;gap:8px;align-items:flex-end;min-width:260px;}'
            '  .nuk-section{min-width:260px;}'
            '  .nuk-section-title{font-size:12px;font-weight:600;color:var(--nuk-fg);margin-bottom:6px;}'
            '  .nuk-legend-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px 12px;}'
            '  .nuk-legend-item{display:flex;align-items:center;gap:8px;min-width:0;}'
            '  .nuk-swatch{width:12px;height:12px;border-radius:3px;border:1px solid rgba(255,255,255,.25);flex:0 0 auto;}'
            '  .nuk-legend-label{font-size:12px;color:var(--nuk-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}'
            '  .nuk-stats{display:flex;gap:12px;flex-wrap:wrap;justify-content:flex-end;font-size:12px;color:var(--nuk-muted);}'
            '  .nuk-help{max-width:520px;font-size:12px;color:var(--nuk-muted);text-align:right;}'
            '  .card{margin:0 !important;border:0 !important;background:transparent !important;}'
            '  .card-body{padding:0 !important;}'
            '  #mynetwork{width:100% !important;height:calc(100vh - var(--nuk-toolbar-h) - var(--nuk-infobar-h)) !important;background:var(--nuk-bg) !important;border:0 !important;}'
            '  body.nuk-hide-info #mynetwork{height:calc(100vh - var(--nuk-toolbar-h)) !important;}'
            '  @media (max-width: 820px){:root{--nuk-infobar-h:160px;} .nuk-info-right{align-items:flex-start;min-width:0;} .nuk-help{text-align:left;} }'
            '</style>'
        )

        js = (
            '<script id="nuk-ui-script">'
            '  (function(){'
            '    function byId(id){return document.getElementById(id);}'
            '    function setPhysicsLabel(enabled){'
            '      var el = byId("nuk-physics-state");'
            '      if(!el) return;'
            '      el.textContent = "Physics: " + (enabled ? "on" : "off");'
            '    }'
            '    function getNetwork(){'
            '      try{ if(typeof network !== "undefined") return network; }catch(e){}'
            '      return null;'
            '    }'
            '    document.addEventListener("DOMContentLoaded", function(){'
            '      var net = getNetwork();'
            f'      var physicsEnabled = {"true" if initial_physics_enabled else "false"};'
            '      setPhysicsLabel(physicsEnabled);'
            '      if(net && net.setOptions) net.setOptions({physics:{enabled: physicsEnabled}});'
            '      var fitBtn = byId("nuk-fit");'
            '      if(fitBtn) fitBtn.addEventListener("click", function(){ if(net && net.fit) net.fit({animation:true}); });'
            '      var centerBtn = byId("nuk-center");'
            '      if(centerBtn) centerBtn.addEventListener("click", function(){'
            '        if(net && net.moveTo) net.moveTo({position:{x:0,y:0},scale:1,animation:true});'
            '      });'
            '      var physicsBtn = byId("nuk-physics");'
            '      if(physicsBtn) physicsBtn.addEventListener("click", function(){'
            '        physicsEnabled = !physicsEnabled;'
            '        setPhysicsLabel(physicsEnabled);'
            '        if(net && net.setOptions) net.setOptions({physics:{enabled: physicsEnabled}});'
            '      });'
            '      var toggleInfoBtn = byId("nuk-toggle-info");'
            '      if(toggleInfoBtn) toggleInfoBtn.addEventListener("click", function(){'
            '        document.body.classList.toggle("nuk-hide-info");'
            '        if(net && net.redraw) net.redraw();'
            '      });'
            '    });'
            '  })();'
            '</script>'
        )

        lower = html.lower()
        head_close = lower.find('</head>')
        if head_close != -1:
            html = html[:head_close] + css + "\n" + html[head_close:]
        else:
            html = css + "\n" + html

        # Recompute indices after modifying the document
        lower = html.lower()
        body_idx = lower.find('<body')
        if body_idx != -1:
            body_end = html.find('>', body_idx)
            if body_end != -1:
                html = html[:body_end + 1] + "\n" + ui_html + "\n" + html[body_end + 1:]
            else:
                html = ui_html + "\n" + html
        else:
            html = ui_html + "\n" + html

        lower = html.lower()
        body_close = lower.rfind('</body>')
        if body_close != -1:
            html = html[:body_close] + js + "\n" + html[body_close:]
        else:
            html = html + "\n" + js

        return html
    
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
        try:
            html = output_path.read_text(encoding='utf-8', errors='ignore')
            html = self._inject_ui_into_html(html, initial_physics_enabled=show_physics)
            output_path.write_text(html, encoding='utf-8')
        except Exception as e:
            logger.warning(f"Could not inject toolbar/legend UI into HTML: {e}")
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
