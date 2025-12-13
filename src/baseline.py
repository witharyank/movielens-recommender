import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load data
ratings = pd.read_csv("../data/ratings.csv")

# Train-test split
train, test = train_test_split(ratings, test_size=0.2, random_state=42)

# Global mean baseline
global_mean = train["rating"].mean()

# Predict using global mean
test["pred_rating"] = global_mean

# Evaluate
rmse = np.sqrt(mean_squared_error(test["rating"], test["pred_rating"]))

print(f"Global Mean Rating: {global_mean:.3f}")
print(f"Baseline RMSE: {rmse:.3f}")
