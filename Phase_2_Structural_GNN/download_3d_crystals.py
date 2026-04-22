import os
import pandas as pd
from mp_api.client import MPRester
import warnings

warnings.filterwarnings("ignore")

# 🔴 YOUR 32-CHARACTER API KEY 🔴
API_KEY = "NH3YtGeXfU6qhzSTdyVRkkaAK1agfUeI"

SAVE_DIR = "cif_data"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def download_crystal_graphs(num_materials=500):
    print("Connecting to Materials Project supercomputers...")
    
    with MPRester(API_KEY) as mpr:
        # THE FIX: Filter by "elasticity" category, but extract the "bulk_modulus" field
        docs = mpr.materials.summary.search(
            is_stable=True, 
            has_props=["elasticity"],
            fields=["material_id", "formula_pretty", "structure", "bulk_modulus"]
        )
        
        print(f"✅ Found {len(docs)} materials. Downloading 3D structures now...")
        
        target_data = []
        saved_count = 0
        
        for doc in docs:
            if saved_count >= num_materials:
                break
                
            mat_id = str(doc.material_id)
            formula = doc.formula_pretty
            
            try:
                bulk_mod = None
                # Extract the bulk modulus from the returned field
                if doc.bulk_modulus is not None:
                    if isinstance(doc.bulk_modulus, dict):
                        bulk_mod = doc.bulk_modulus.get("vrh")
                    else:
                        bulk_mod = float(doc.bulk_modulus)
                        
                if bulk_mod is None:
                    continue
            except:
                continue 
                
            try:
                # Extract the 3D structure and save it
                structure = doc.structure
                if structure is None:
                    continue 
                    
                cif_filename = f"{mat_id}.cif"
                cif_path = os.path.join(SAVE_DIR, cif_filename)
                structure.to(filename=cif_path, fmt="cif")
                
                # Record the answers
                target_data.append({
                    "material_id": mat_id,
                    "formula": formula,
                    "bulk_modulus_gpa": round(bulk_mod, 2)
                })
                saved_count += 1
                
                if saved_count % 100 == 0:
                    print(f"   ...Downloaded {saved_count} structures...")
            except Exception as e:
                continue

        # Save to CSV
        df = pd.DataFrame(target_data)
        df.to_csv("gnn_targets.csv", index=False)
        
        print("\n🎉 SUCCESS!")
        print(f"Saved {saved_count} 3D .cif files into the '{SAVE_DIR}' folder.")
        print("Saved the GNN answer key to 'gnn_targets.csv'.")

if __name__ == "__main__":
    download_crystal_graphs(num_materials=1500)