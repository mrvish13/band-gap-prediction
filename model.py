import pandas as pd
from mp_api.client import MPRester
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

API_KEY = "8rfUzG7tGOPUpd7krKfwgYsoOOFqWHbJ"

# Pull data
with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        elements=["Fe", "O"],
        fields=["material_id", "formula_pretty", "band_gap", 
                "formation_energy_per_atom", "density", "volume"]
    )

# Build dataframe
rows = []
for doc in docs:
    rows.append({
        "formula": doc.formula_pretty,
        "band_gap": doc.band_gap,
        "formation_energy": doc.formation_energy_per_atom,
        "density": doc.density,
        "volume": doc.volume
    })

df = pd.DataFrame(rows).dropna()

# Features and target
X = df[["formation_energy", "density", "volume"]]
y = df["band_gap"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.3f} eV")

# Plot
plt.scatter(y_test, y_pred, alpha=0.4)
plt.xlabel("Actual Band Gap (eV)")
plt.ylabel("Predicted Band Gap (eV)")
plt.title("Random Forest: Predicted vs Actual Band Gap")
plt.savefig("band_gap_prediction.png")
print("\nPlot saved as band_gap_prediction.png")