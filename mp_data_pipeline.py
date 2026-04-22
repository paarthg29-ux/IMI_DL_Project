# Install the official client first: pip install mp-api pandas
from mp_api.client import MPRester
import pandas as pd

class MaterialsProjectPipeline:
    def __init__(self, api_key):
        """
        Initializes the production connection to The Materials Project.
        Requires a valid API key from materialsproject.org.
        """
        self.api_key = 'NH3YtGeXfU6qhzSTdyVRkkaAK1agfUeI'
        self.dataset = None

    def fetch_training_data(self):
        """
        Queries the MP database for materials that specifically have 
        elasticity data (strength) and density.
        """
        print("Connecting to The Materials Project API...")
        
        # Open a connection using the modern MPRester
        with MPRester(self.api_key) as mpr:
            # We search the 'summary' endpoint for materials that have elasticity calculated.
            # We only pull the specific fields we need to save bandwidth and memory.
            results = mpr.materials.summary.search(
                has_props=["elasticity"],
                fields=["formula_pretty", "density", "bulk_modulus", "shear_modulus"]
            )
        
        print(f"Successfully downloaded {len(results)} materials.")
        
        # Convert the complex API objects into a clean Pandas DataFrame for ML
        data_list = []
        for doc in results:
            data_list.append({
                "Formula": doc.formula_pretty,
                "Density_g_cm3": doc.density,
                "Bulk_Modulus_GPa": doc.bulk_modulus.get('voigt', None) if doc.bulk_modulus else None,
                "Shear_Modulus_GPa": doc.shear_modulus.get('voigt', None) if doc.shear_modulus else None
            })
            
        self.dataset = pd.DataFrame(data_list).dropna()
        print(f"Cleaned dataset ready for training: {len(self.dataset)} rows.")
        
        return self.dataset

# --- Execution for Backend ---
if __name__ == "__main__":
    # Replace 'YOUR_API_KEY' with your actual Materials Project API key
    MP_API_KEY = "YOUR_API_KEY" 
    
    pipeline = MaterialsProjectPipeline(MP_API_KEY)
    
    try:
        df = pipeline.fetch_training_data()
        print("\nHead of Final Dataset:")
        print(df.head())
    except Exception as e:
        print(f"Failed to fetch data: {e}")
        print("Make sure your API key is correct and active.")