import pandas as pd
import numpy as np
import os

DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

n_samples = 500  # you can increase for better accuracy

# Generate synthetic dataset
data = {
    "Area": np.random.randint(500, 5000, size=n_samples),
    "Bedrooms": np.random.randint(1, 10, size=n_samples),
    "Bathrooms": np.random.randint(1, 10, size=n_samples),
    "Garage": np.random.randint(0, 5, size=n_samples),
    "Grade": np.random.randint(1, 10, size=n_samples),
    "LotArea": np.random.randint(1000, 10000, size=n_samples),
    "Location": np.random.choice(["Downtown", "Suburb", "Countryside", "Beachfront"], size=n_samples)
}

# Simple target formula (for synthetic data)
df = pd.DataFrame(data)
df["Price"] = (df["Area"]*200 + df["Bedrooms"]*10000 + df["Bathrooms"]*8000 +
               df["Garage"]*5000 + df["Grade"]*15000 + df["LotArea"]*10 +
               np.random.randint(-50000, 50000, size=n_samples))

# Save CSV
df.to_csv(os.path.join(DATA_PATH, "housing.csv"), index=False)
print(f"Generated {n_samples} rows of housing data at {DATA_PATH}/housing.csv")
