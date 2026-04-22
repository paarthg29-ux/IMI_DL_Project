import torch
import pandas as pd
import numpy as np
import joblib
import time
from train_vae import MaterialVAE

class AI_Material_Discoverer:
    def __init__(self, predictor_path="property_predictor.joblib", vae_path="material_vae.pth", train_data="engineered_materials.csv"):
        self.screener_data = joblib.load(predictor_path)
        self.screener_model = self.screener_data['model']
        self.features = self.screener_data['features']
        
        try:
            self.training_formulas = set(pd.read_csv(train_data)['Formula'].dropna().unique())
        except:
            self.training_formulas = set()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = MaterialVAE(input_dim=len(self.features), latent_dim=4).to(self.device)
        self.generator.load_state_dict(torch.load(vae_path, map_location=self.device, weights_only=True))
        self.generator.eval()

    def invent_materials(self, num_samples=1000):
        with torch.no_grad():
            z = (torch.randn(num_samples, 4) * 3.0).to(self.device)
            generated_raw = self.generator.decode(z).cpu().numpy()
        return self._clean_compositions(generated_raw)

    def _clean_compositions(self, raw_fractions):
        clean_data = []
        for row in raw_fractions:
            row[row < 0.01] = 0.0
            if np.sum(row) > 0:
                row = row / np.sum(row)
            clean_data.append(row)
        df = pd.DataFrame(clean_data, columns=self.features)
        return df[(df.T != 0).any()].copy()

    def screen_candidates(self, candidates_df):
        predictions = self.screener_model.predict(candidates_df)
        candidates_df = candidates_df.copy()
        candidates_df['Predicted_Density'] = predictions[:, 0]
        candidates_df['Predicted_Bulk_Modulus_GPa'] = predictions[:, 1]

        # ── UI HOOK RESTORED: Save ALL screened results for Streamlit histograms ──
        candidates_df[['Predicted_Density', 'Predicted_Bulk_Modulus_GPa']].to_csv(
            "all_screened_candidates.csv", index=False
        )

        return candidates_df.sort_values(by='Predicted_Bulk_Modulus_GPa', ascending=False)

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
                "Density": row['Predicted_Density'],
            })
        # ── UI HOOK RESTORED ──
        pd.DataFrame(results).to_csv("top_materials.csv", index=False)
        return results

    def extract_formulas(self, df):
        formulas = []
        for index, row in df.iterrows():
            formula = ""
            for elem in self.features:
                if row[elem] > 0:
                    formula += f"{elem}{row[elem]:.2f} "
            formulas.append(formula.strip())
        return formulas

    def test_pipeline(self, test_size=10000, high_strength_threshold=150.0):
        print(f"\n--- INITIATING FULL PIPELINE STRESS TEST ({test_size} Samples) ---")
        start_gen = time.time()
        
        # Raw generation to calculate validity
        with torch.no_grad():
            z = (torch.randn(test_size, 4) * 3.0).to(self.device)
            generated_raw = self.generator.decode(z).cpu().numpy()
            
        candidates_df = self._clean_compositions(generated_raw)
        gen_time = time.time() - start_gen
        
        validity_rate = len(candidates_df) / test_size * 100
        
        start_screen = time.time()
        screened_results = self.screen_candidates(candidates_df)
        screen_time = time.time() - start_screen
        
        formulas = self.extract_formulas(screened_results)
        novel_count = sum(1 for f in formulas if f not in self.training_formulas)
        novelty_rate = novel_count / len(formulas) * 100 if formulas else 0
        
        high_strength_count = len(screened_results[screened_results['Predicted_Bulk_Modulus_GPa'] >= high_strength_threshold])
        
        print(f"⚡ Pipeline Benchmarks:")
        print(f"  • Generation Time:  {gen_time:.2f} seconds")
        print(f"  • Screening Time:   {screen_time:.2f} seconds")
        print(f"📊 Quality Metrics:")
        print(f"  • Validity Rate:    {validity_rate:.1f}%")
        print(f"  • Novelty Rate:     {novelty_rate:.1f}%")
        print(f"💎 Yield Metrics:")
        print(f"  • High-Strength Candidates (> {high_strength_threshold} GPa): {high_strength_count} discovered")
        print("----------------------------------------------------------\n")

if __name__ == "__main__":
    discoverer = AI_Material_Discoverer()
    discoverer.test_pipeline(test_size=10000)