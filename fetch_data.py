import pandas as pd
from mp_api.client import MPRester

API_KEY = "8rfUzG7tGOPUpd7krKfwgYsoOOFqWHbJ"

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        elements=["Fe", "O"],
        fields=["material_id", "formula_pretty", "band_gap", "formation_energy_per_atom"]
    )

rows = []
for doc in docs:
    rows.append({
        "material_id": str(doc.material_id),
        "formula": doc.formula_pretty,
        "band_gap": doc.band_gap,
        "formation_energy": doc.formation_energy_per_atom
    })

df = pd.DataFrame(rows)

print(df.head(10))
print(f"\nTotal materials: {len(df)}")
print(f"\nBand gap stats:\n{df['band_gap'].describe()}")