import torch
import pandas as pd
import numpy as np
import joblib
from train_vae import MaterialVAE

class AI_Material_Discoverer:
    def __init__(self, predictor_path="property_predictor.joblib", vae_path="material_vae.pth"):
        print("1. Loading AI Models...")
        self.screener_data = joblib.load(predictor_path)
        self.screener_model = self.screener_data['model']
        self.features = self.screener_data['features']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = MaterialVAE(input_dim=len(self.features), latent_dim=4).to(self.device)
        self.generator.load_state_dict(torch.load(vae_path, map_location=self.device, weights_only=True))
        self.generator.eval() 

    def invent_materials(self, num_samples=1000):
        print(f"2. Inventing {num_samples} brand new material compositions...")
        with torch.no_grad(): 
            # THE FIX: We multiply the random noise by 3.0 (Temperature). 
            # This forces the AI to hallucinate wildly different elements instead of playing it safe.
            z = (torch.randn(num_samples, 4) * 3.0).to(self.device)
            
            generated_raw = self.generator.decode(z).cpu().numpy()
            
        return self._clean_compositions(generated_raw)

    def _clean_compositions(self, raw_fractions):
        clean_data = []
        for row in raw_fractions:
            # THE FIX: We lowered the threshold to 1% (0.01) so we don't accidentally delete trace metals
            row[row < 0.01] = 0.0
            
            if np.sum(row) > 0:
                row = row / np.sum(row)
            clean_data.append(row)
            
        df = pd.DataFrame(clean_data, columns=self.features)
        return df[(df.T != 0).any()]

    def screen_candidates(self, candidates_df):
        print("3. Screening candidates through the Property Predictor...")
        predictions = self.screener_model.predict(candidates_df)
        
        candidates_df['Predicted_Density'] = predictions[:, 0]
        candidates_df['Predicted_Bulk_Modulus_GPa'] = predictions[:, 1]
        
        # We sort to find the strongest materials
        best_materials = candidates_df.sort_values(by='Predicted_Bulk_Modulus_GPa', ascending=False)
        return best_materials

    def format_top_materials(self, best_materials, top_n=5):
        results = []
        for index, row in best_materials.head(top_n).iterrows():
            formula = ""
            for elem in self.features:
                fraction = row[elem]
                if fraction > 0:
                    formula += f"{elem}{fraction:.2f} "
            
            results.append({
                "Formula": formula.strip(),
                "Strength_GPa": row['Predicted_Bulk_Modulus_GPa'],
                "Density": row['Predicted_Density']
            })
        return results

if __name__ == "__main__":
    discoverer = AI_Material_Discoverer()
    candidates = discoverer.invent_materials(num_samples=5000)
    screened_results = discoverer.screen_candidates(candidates)
    top_5 = discoverer.format_top_materials(screened_results, top_n=5)
    
    print("\n🏆 Top 5 AI-Discovered Materials 🏆")
    for i, mat in enumerate(top_5, 1):
        print(f"{i}. {mat['Formula']} | Predicted Strength: {mat['Strength_GPa']:.1f} GPa | Predicted Density: {mat['Density']:.2f} g/cm³")