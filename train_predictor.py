import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

class PropertyPredictor:
    def __init__(self, data_file="engineered_materials.csv"):
        self.data_file = data_file
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        # We will save the exact names of our 88 elemental features so the future 
        # dashboard knows exactly what inputs to provide.
        self.feature_columns = None

    def train_and_save(self, model_filename="property_predictor.joblib"):
        print(f"1. Loading engineered dataset from {self.data_file}...")
        df = pd.read_csv(self.data_file)
        
        # Drop rows where our target variables are missing just to be safe
        df = df.dropna(subset=['Density', 'Bulk_Modulus'])

        # 'y' is what we want to predict (The targets)
        y = df[['Density', 'Bulk_Modulus']]
        
        # 'X' is what we use to make the prediction (The 88 elemental fractions)
        # We drop the original text formula and the target columns to isolate just the elements
        X = df.drop(columns=['Formula', 'Density', 'Bulk_Modulus', 'Shear_Modulus'])
        self.feature_columns = X.columns.tolist()

        print("2. Splitting data into Training (80%) and Testing (20%) sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("3. Training the Random Forest AI Model (this might take a few seconds)...")
        self.model.fit(X_train, y_train)

        print("4. Testing the AI on unseen data...")
        predictions = self.model.predict(X_test)
        
        # Evaluate how far off the AI's guesses are on average
        mae_density = mean_absolute_error(y_test['Density'], predictions[:, 0])
        mae_bulk = mean_absolute_error(y_test['Bulk_Modulus'], predictions[:, 1])
        
        print("\n--- AI Model Performance ---")
        print(f"Average Error for Density: ±{mae_density:.2f} g/cm³")
        print(f"Average Error for Bulk Modulus (Strength): ±{mae_bulk:.2f} GPa")
        print("----------------------------\n")

        print("5. Saving the trained AI and feature list for the final backend...")
        # We package the trained model AND the list of feature names together
        # so the final web app knows exactly how to format user inputs.
        package = {
            'model': self.model,
            'features': self.feature_columns
        }
        joblib.dump(package, model_filename)
        print(f"Success! Model securely saved as '{model_filename}'.")

if __name__ == "__main__":
    trainer = PropertyPredictor()
    trainer.train_and_save()