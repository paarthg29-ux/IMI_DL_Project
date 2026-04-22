import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

class PropertyPredictor:
    def __init__(self, data_file="engineered_materials.csv"):
        self.data_file = data_file
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.feature_columns = None

    def train_and_save(self, model_filename="property_predictor.joblib"):
        print("1. Loading dataset...")
        df = pd.read_csv(self.data_file).dropna(subset=['Density', 'Bulk_Modulus'])

        y = df[['Density', 'Bulk_Modulus']]
        X = df.drop(columns=['Formula', 'Density', 'Bulk_Modulus', 'Shear_Modulus'])
        self.feature_columns = X.columns.tolist()

        print("2. Splitting and Training...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        print("3. Evaluating Metrics...")
        preds = self.model.predict(X_test)
        
        # --- NEW: ADVANCED METRICS ---
        metrics = {
            "Density": {
                "MAE": mean_absolute_error(y_test['Density'], preds[:, 0]),
                "RMSE": np.sqrt(mean_squared_error(y_test['Density'], preds[:, 0])),
                "R2": r2_score(y_test['Density'], preds[:, 0])
            },
            "Bulk_Modulus": {
                "MAE": mean_absolute_error(y_test['Bulk_Modulus'], preds[:, 1]),
                "RMSE": np.sqrt(mean_squared_error(y_test['Bulk_Modulus'], preds[:, 1])),
                "R2": r2_score(y_test['Bulk_Modulus'], preds[:, 1])
            }
        }
        
        print("\n--- Model Performance ---")
        for target, scores in metrics.items():
            print(f"[{target}] MAE: ±{scores['MAE']:.2f} | RMSE: ±{scores['RMSE']:.2f} | R²: {scores['R2']:.4f}")
            
        # --- NEW: ACTUAL VS PREDICTED SCATTER PLOTS ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.scatter(y_test['Density'], preds[:, 0], alpha=0.3, color='blue')
        ax1.plot([y_test['Density'].min(), y_test['Density'].max()], [y_test['Density'].min(), y_test['Density'].max()], 'k--', lw=2)
        ax1.set_title(f"Density: Actual vs Predicted (R² = {metrics['Density']['R2']:.2f})")
        ax1.set_xlabel("Actual Density")
        ax1.set_ylabel("Predicted Density")

        ax2.scatter(y_test['Bulk_Modulus'], preds[:, 1], alpha=0.3, color='red')
        ax2.plot([y_test['Bulk_Modulus'].min(), y_test['Bulk_Modulus'].max()], [y_test['Bulk_Modulus'].min(), y_test['Bulk_Modulus'].max()], 'k--', lw=2)
        ax2.set_title(f"Bulk Modulus: Actual vs Predicted (R² = {metrics['Bulk_Modulus']['R2']:.2f})")
        ax2.set_xlabel("Actual Bulk Modulus (GPa)")
        ax2.set_ylabel("Predicted Bulk Modulus (GPa)")

        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')

        # --- FEATURE IMPORTANCE BAR CHARTS ---
        self._plot_feature_importance()

        joblib.dump({'model': self.model, 'features': self.feature_columns}, model_filename)
        print("\nSuccess! Saved 'actual_vs_predicted.png', 'feature_importance.png', and model.")

    def _plot_feature_importance(self):
        """Train two single-output RF models to get per-target feature importances,
        then save a side-by-side Top-15 horizontal bar chart."""
        print("4. Computing Feature Importances...")

        df = pd.read_csv(self.data_file).dropna(subset=['Density', 'Bulk_Modulus'])
        X = df.drop(columns=['Formula', 'Density', 'Bulk_Modulus', 'Shear_Modulus'])
        y_density = df['Density']
        y_bulk    = df['Bulk_Modulus']

        # Two lightweight single-output models (50 trees — fast, still representative)
        rf_density = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_bulk    = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        rf_density.fit(X, y_density)
        rf_bulk.fit(X, y_bulk)

        # Top-15 importances, sorted ascending so longest bar is at top
        def top15(model, columns):
            imp = pd.Series(model.feature_importances_, index=columns)
            return imp.nlargest(15).sort_values()

        imp_density = top15(rf_density, self.feature_columns)
        imp_bulk    = top15(rf_bulk,    self.feature_columns)

        # ── Plot ────────────────────────────────────────────────────────────────
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.patch.set_facecolor('#0D1117')   # dark background — looks great in PPT

        def style_ax(ax, importances, color, title):
            bars = ax.barh(importances.index, importances.values,
                           color=color, edgecolor='none', height=0.65)
            # Value label at the end of each bar
            for bar, val in zip(bars, importances.values):
                ax.text(bar.get_width() + 0.0005,
                        bar.get_y() + bar.get_height() / 2,
                        f'{val:.4f}', va='center', ha='left',
                        fontsize=9, color='white')
            ax.set_facecolor('#161B22')
            ax.set_title(title, fontsize=13, fontweight='bold', color=color, pad=12)
            ax.set_xlabel('Importance score (mean decrease in impurity)',
                          fontsize=10, color='#8B9BC1')
            ax.tick_params(colors='white', labelsize=10)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.spines['left'].set_color('#2D333B')
            ax.spines['bottom'].set_color('#2D333B')
            ax.xaxis.label.set_color('#8B9BC1')
            ax.set_xlim(0, importances.values.max() * 1.25)  # room for labels

        style_ax(ax1, imp_density, '#00C8FF', 'Top 15 elements — Density prediction')
        style_ax(ax2, imp_bulk,    '#F5C518', 'Top 15 elements — Bulk Modulus prediction')

        fig.suptitle('Random Forest — Feature Importance (elemental fractions)',
                     fontsize=14, fontweight='bold', color='white', y=1.01)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150,
                    bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print("   Saved 'feature_importance.png'")

if __name__ == "__main__":
    trainer = PropertyPredictor()
    trainer.train_and_save()