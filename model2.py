import pandas as pd
from mp_api.client import MPRester
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

API_KEY = "8rfUzG7tGOPUpd7krKfwgYsoOOFqWHbJ"

if __name__ == '__main__':
    # Pull data
    print("Fetching data...")
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            elements=["Fe", "O"],
            fields=["material_id", "formula_pretty", "band_gap",
                    "formation_energy_per_atom", "density"]
        )

    # Build dataframe
    rows = []
    for doc in docs:
        rows.append({
            "formula": doc.formula_pretty,
            "band_gap": doc.band_gap,
            "formation_energy": doc.formation_energy_per_atom,
            "density": doc.density
        })

    df = pd.DataFrame(rows).dropna()
    print(f"Got {len(df)} materials")

    # Generate smart chemistry features using matminer
    print("Generating features...")
    ep = ElementProperty.from_preset("magpie")
    df["composition"] = df["formula"].apply(lambda x: Composition(x))
    df = ep.featurize_dataframe(df, col_id="composition", ignore_errors=True)
    df = df.dropna()

    # Features and target
    drop_cols = ["formula", "band_gap", "composition"]
    X = df.drop(columns=drop_cols)
    y = df["band_gap"]

    # Train model
    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\nR² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f} eV")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.plot([0, 5], [0, 5], "r--", label="Perfect prediction")
    plt.xlabel("Actual Band Gap (eV)")
    plt.ylabel("Predicted Band Gap (eV)")
    plt.title("Random Forest with Matminer Features")
    plt.legend()
    plt.savefig("band_gap_v2.png")
    print("Plot saved as band_gap_v2.png")

# Feature importance plot
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top20 = importances.sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 7))
    top20.plot(kind='bar')
    plt.title("Top 20 Most Important Features for Band Gap Prediction")
    plt.ylabel("Importance Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("Feature importance plot saved!")