# src/analyzer.py
import prolif as plf
import pandas as pd
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class InteractionEngine:
    def __init__(self):
        self.fp = plf.Fingerprint([
            'HBDonor', 
            'HBAcceptor', 
            'PiStacking', 
            'Hydrophobic', 
            'VdWContact'
        ])

    def run_analysis(self, universe, ligand_ag, target_ag):
        print(f"⚙️  Analyzing interactions between Ligand ({ligand_ag.n_residues} res) and Targets ({target_ag.n_residues} res)...")

        # 1. Define Settings
        # We need force=True to ignore missing hydrogens.
        common_settings = {"force": True, "NoImplicit": False}
        
        # 2. Map settings to molecules
        # Index 0 = Ligand, Index 1 = Protein/DNA
        safe_kwargs = {0: common_settings, 1: common_settings}

        try:
            # 3. Run Analysis
            # We pass safe_kwargs so ProLIF knows how to handle EACH molecule
            self.fp.run(
                universe.trajectory, 
                ligand_ag, 
                target_ag,
                converter_kwargs=safe_kwargs, # <--- FIXED STRUCTURE
                progress=False
            )
        except AttributeError:
            # Fallback for single-frame issues
            print("⚠️  Parallel run failed. Switching to single-threaded mode...")
            try:
                # Manual run for single frame
                lig_mol = plf.Molecule.from_mda(ligand_ag, **common_settings)
                prot_mol = plf.Molecule.from_mda(target_ag, **common_settings)
                self.fp.generate(lig_mol, prot_mol, residues=None)
            except Exception as e:
                print(f"❌ Analysis failed: {e}")
                return pd.DataFrame()

        # 4. Extract Data
        try:
            df_raw = self.fp.to_dataframe()
        except:
            if hasattr(self.fp, "ifp"):
                # Handle single-frame dict output
                data = {0: self.fp.ifp} 
                df_raw = pd.DataFrame.from_dict(data, orient='index')
            else:
                return pd.DataFrame()
        
        if df_raw.empty:
            print("⚠️  No interactions detected.")
            return pd.DataFrame()

        clean_df = self._melt_results(df_raw)
        return clean_df

    def _melt_results(self, df):
        """
        Flattens the complex ProLIF dataframe into a simple CSV format.
        """
        # Ensure Frame index exists
        if "Frame" not in df.index.names and "Frame" not in df.columns:
            df["Frame"] = 0
            
        df = df.reset_index()
        melted = []
        
        for col in df.columns:
            # Skip metadata columns
            if col == "Frame" or col == "index": continue
            
            # ProLIF output is MultiIndex-like tuple: (LigandRes, TargetRes, Interaction)
            if isinstance(col, tuple):
                ligand_res = str(col[0])
                target_res = str(col[1])
                interaction = str(col[2])
                
                # Value is Boolean (True/False)
                value = df[col].values[0]
                
                if value: 
                    melted.append({
                        "Ligand": ligand_res,
                        "Target": target_res,
                        "Interaction": interaction,
                        "Distance": "Detected"
                    })
                
        return pd.DataFrame(melted)