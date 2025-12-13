from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Load ratings
ratings = pd.read_csv("../data/ratings.csv")

# Surprise needs only userId, movieId, rating
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD model
model = SVD(random_state=42)
model.fit(trainset)

# Evaluate
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)

print(f"SVD RMSE: {rmse:.3f}")
