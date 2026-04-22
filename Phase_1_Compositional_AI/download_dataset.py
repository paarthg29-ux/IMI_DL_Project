# Make sure you have the required libraries installed in your terminal:
# pip install mp-api pandas

from mp_api.client import MPRester
import pandas as pd

class MaterialsDataDownloader:
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_and_save(self, filename="materials_data.csv"):
        print("Connecting to The Materials Project API...")
        
        # Open a secure connection using your API key
        with MPRester(self.api_key) as mpr:
            # Search for materials that specifically have elasticity (strength) data
            results = mpr.materials.summary.search(
                has_props=["elasticity"],
                fields=["formula_pretty", "density", "bulk_modulus", "shear_modulus"]
            )
        
        print(f"Successfully downloaded {len(results)} materials.")
        
        # Extract the exact data points we need into a list
        data_list = []
        for doc in results:
            data_list.append({
                "Formula": doc.formula_pretty,
                "Density": doc.density,
                "Bulk_Modulus": doc.bulk_modulus.get('voigt', None) if doc.bulk_modulus else None,
                "Shear_Modulus": doc.shear_modulus.get('voigt', None) if doc.shear_modulus else None
            })
            
        # Convert to a data table and remove any rows with missing information
        df = pd.DataFrame(data_list).dropna()
        
        # Save permanently to your local folder
        df.to_csv(filename, index=False)
        print(f"Dataset securely saved to '{filename}' with {len(df)} ready-to-use materials.")

# Execute the download
if __name__ == "__main__":
    API_KEY = "NH3YtGeXfU6qhzSTdyVRkkaAK1agfUeI" 
    downloader = MaterialsDataDownloader(API_KEY)
    downloader.fetch_and_save()