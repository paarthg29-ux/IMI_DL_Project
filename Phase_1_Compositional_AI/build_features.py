import pandas as pd
from pymatgen.core import Composition

class FeatureEngineer:
    def __init__(self, input_file="materials_data.csv", output_file="engineered_materials.csv"):
        """
        Initializes the feature engineering pipeline.
        Reads the local CSV so we don't have to query the API again.
        """
        self.input_file = input_file
        self.output_file = output_file

    def process_features(self):
        print(f"1. Loading raw data from {self.input_file}...")
        try:
            df = pd.read_csv(self.input_file)
        except FileNotFoundError:
            print(f"Error: Could not find {self.input_file}. Make sure you are running this in the correct folder.")
            return

        print(f"Loaded {len(df)} materials.")
        print("2. Starting Feature Engineering (converting formulas to atomic numbers)...")
        
        # This list will hold a dictionary of elements for every single material
        element_fractions = []
        
        for index, row in df.iterrows():
            formula = row['Formula']
            try:
                # pymatgen does the hard work of understanding chemistry syntax
                comp = Composition(formula)
                element_fractions.append(comp.fractional_composition.as_dict())
            except Exception as e:
                # If a formula is weirdly formatted, we log an empty dict
                element_fractions.append({})

        # Convert the list of dictionaries into a DataFrame where each element gets its own column
        features_df = pd.DataFrame(element_fractions).fillna(0.0)
        
        # Combine the original targets (Density, Modulus) with our new numeric features
        final_dataset = pd.concat([df, features_df], axis=1)
        
        print("3. Saving engineered dataset...")
        # Save to CSV permanently
        final_dataset.to_csv(self.output_file, index=False)
        
        print(f"Success! Saved completely engineered dataset to {self.output_file}.")
        print(f"The AI now has {len(features_df.columns)} unique elemental features to learn from.")

if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.process_features()