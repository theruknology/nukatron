# main.py
from src.loader import StructureLoader
from src.analyzer import InteractionEngine
import pandas as pd

# --- CONFIGURATION ---
PDB_FILE = "data/complex.pdb"  # <--- REPLACE WITH YOUR FILE PATH
LIGAND_CODE = "MBC"            # <--- REPLACE WITH YOUR 3-LETTER DRUG CODE
OUTPUT_FILE = "output/results.csv"

def main():
    # 1. Init Loader
    print("--- 🚀 Starting Phase 1 ---")
    loader = StructureLoader(PDB_FILE)
    
    # 2. Get Atoms
    # Note: If your PDB uses a different code for the drug (e.g., DRG, INH), change LIGAND_CODE above.
    ligand = loader.get_ligand(LIGAND_CODE)
    targets = loader.get_targets() # Gets DNA + Protein

    # 3. Run Analysis
    engine = InteractionEngine()
    df = engine.run_analysis(loader.u, ligand, targets)

    # 4. Save
    if not df.empty:
        print("\n--- ✅ SUCCESS: Interactions Found ---")
        print(df[["Target", "Interaction"]].to_string())
        
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n📄 Saved full report to: {OUTPUT_FILE}")
    else:
        print("\n--- ❌ No interactions found. Check distance cutoffs or PDB quality. ---")

if __name__ == "__main__":
    main()