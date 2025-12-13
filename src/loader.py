# src/loader.py
import MDAnalysis as mda
import warnings

# Suppress MDAnalysis warnings about missing chain IDs or temp factors
warnings.filterwarnings("ignore", category=UserWarning)

class StructureLoader:
    def __init__(self, pdb_path):
        """
        Initializes the loader with a PDB file.
        1. Loads the structure.
        2. STRIPS hydrogens (they cause bond-guessing errors).
        3. Guesses bonds on heavy atoms only.
        """
        self.pdb_path = pdb_path
        try:
            # 1. Load Raw
            raw_u = mda.Universe(pdb_path)
            
            # 2. Strip Hydrogens to fix "KekulizeException"
            # RDKit crashes if PDB hydrogens conflict with guessed bonds.
            # We keep only "Heavy Atoms" (not H).
            heavy_atoms = raw_u.select_atoms("not element H")
            
            print(f"🧹 Cleaning structure: Dropping {len(raw_u.atoms) - len(heavy_atoms)} Hydrogen atoms...")
            
            # Create a new clean Universe with only heavy atoms
            self.u = mda.Merge(heavy_atoms)
            print(f"✅ Loaded Clean Universe: {len(self.u.atoms)} atoms")
            
            # 3. Guess Bonds (Critical step)
            # Now that H is gone, this works much better.
            if not hasattr(self.u.atoms, 'bonds') or len(self.u.atoms.bonds) == 0:
                print("⚠️  No bond info found in PDB. Guessing bonds based on distance...")
                self.u.atoms.guess_bonds()
                
        except Exception as e:
            raise ValueError(f"Failed to load structure: {e}")

    def get_ligand(self, resname):
        """
        Selects the ligand by 3-letter residue name.
        """
        ligand = self.u.select_atoms(f"resname {resname}")
        
        if ligand.n_atoms == 0:
            # Fallback: Try identifying by residue number if name fails
            # But usually raising error is safer
            raise ValueError(f"❌ Could not find ligand with residue name '{resname}'")
            
        return ligand

    def get_targets(self):
        """
        Selects everything that IS NOT water and NOT a dummy atom.
        Targeting Protein AND Nucleic acids.
        """
        targets = self.u.select_atoms("protein or nucleic")
        
        if targets.n_atoms == 0:
            raise ValueError("❌ No Protein or DNA found in the structure.")
            
        return targets