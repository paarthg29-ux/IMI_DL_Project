import os
import pandas as pd
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data
import warnings

warnings.filterwarnings("ignore")

TARGETS_FILE = "gnn_targets.csv"
CIF_DIR = "cif_data"
OUTPUT_FILE = "crystal_graphs_dataset.pt"

df = pd.read_csv(TARGETS_FILE)
graph_dataset = []

print(f"Converting {len(df)} 3D .cif files into PyTorch Neural Graphs...")

success_count = 0
dropped_count = 0

for index, row in df.iterrows():
    mat_id = row['material_id']
    target_modulus = row['bulk_modulus_gpa']
    cif_path = os.path.join(CIF_DIR, f"{mat_id}.cif")
    
    if not os.path.exists(cif_path):
        continue

    try:
        crystal = Structure.from_file(cif_path)
        
        # Drop overly massive crystals to save RAM
        if len(crystal) > 150:
            dropped_count += 1
            continue
            
        # --- THE 12-TRAIT PHYSICS ENGINE ---
        node_features = []
        for site in crystal:
            el = site.specie 
            
            z = float(el.Z)                                  
            mass = float(el.atomic_mass)                     
            en = float(el.X) if el.X else 0.0                
            rad = float(el.atomic_radius) if el.atomic_radius else 0.0 
            group = float(el.group)                          
            period = float(el.row)                           
            mendeleev = float(el.mendeleev_no)               
            ion_energy = float(el.ionization_energies[0]) if el.ionization_energies else 0.0
            elec_affinity = float(el.electron_affinity) if el.electron_affinity else 0.0
            max_ox = float(max(el.common_oxidation_states)) if el.common_oxidation_states else 0.0
            
            # The 2 New Super-Traits
            molar_vol = float(el.molar_volume) if el.molar_volume else 0.0
            melt_pt = float(el.melting_point) if el.melting_point else 0.0
            
            node_features.append([z, mass, en, rad, group, period, mendeleev, ion_energy, elec_affinity, max_ox, molar_vol, melt_pt])
            
        x = torch.tensor(node_features, dtype=torch.float)
        
        # --- WIDENED 4.0A SPATIAL NET ---
        all_neighbors = crystal.get_all_neighbors(r=4.0)
        
        source_nodes = []
        target_nodes = []
        edge_distances = []
        
        for i, neighbor_list in enumerate(all_neighbors):
            for neighbor in neighbor_list:
                source_nodes.append(i)               
                target_nodes.append(neighbor.index)  
                edge_distances.append([neighbor.nn_distance]) 

        if len(source_nodes) == 0:
            dropped_count += 1
            continue

        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        edge_attr = torch.tensor(edge_distances, dtype=torch.float)
        y = torch.tensor([[target_modulus]], dtype=torch.float)

        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graph_dataset.append(graph)
        success_count += 1

    except Exception as e:
        dropped_count += 1

print("\n--- DATA PROCESSING COMPLETE ---")
print(f"✅ Successfully created {success_count} perfect 12-Trait 3D Graphs.")
print(f"🗑️ Dropped {dropped_count} massive or corrupted crystals.")

torch.save(graph_dataset, OUTPUT_FILE)
print(f"Saved all graphs to '{OUTPUT_FILE}'")