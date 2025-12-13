import pandas as pd
from surprise import Dataset, Reader, SVD

# Load data
ratings = pd.read_csv("../data/ratings.csv")
movies = pd.read_csv("../data/movies.csv")

# Train SVD model on full data
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

trainset = data.build_full_trainset()
model = SVD(random_state=42)
model.fit(trainset)

# Recommendation function
def recommend_movies(user_id, n=10):
    seen_movies = set(ratings[ratings["userId"] == user_id]["movieId"])
    all_movies = set(movies["movieId"])

    unseen_movies = all_movies - seen_movies

    predictions = []
    for movie_id in unseen_movies:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]

    # Map movie IDs to titles
    results = []
    for movie_id, rating in top_n:
        title = movies[movies["movieId"] == movie_id]["title"].values[0]
        results.append((title, round(rating, 2)))

    return results

# Test recommendation
user_id = 1
recommendations = recommend_movies(user_id)

print(f"Top recommendations for User {user_id}:")
for title, rating in recommendations:
    print(f"{title} (predicted rating: {rating})")
