import pandas as pd
import joblib
import os
from surprise import Dataset, Reader, SVD

MODEL_PATH = "svd_model.pkl"


# ---------------------------
# Data Loading
# ---------------------------
def load_data():
    ratings = pd.read_csv("../data/ratings.csv")
    movies = pd.read_csv("../data/movies.csv")

    movie_titles = dict(zip(movies.movieId, movies.title))

    return ratings, movies, movie_titles


# ---------------------------
# Train Model
# ---------------------------
def train_model(ratings, retrain: bool = False, random_state: int = 42) -> SVD:
    if os.path.exists(MODEL_PATH) and not retrain:
        print("Loading existing model...")
        return joblib.load(MODEL_PATH)

    print("Training new model...")

    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()

    model = SVD(random_state=random_state)
    model.fit(trainset)

    joblib.dump(model, MODEL_PATH)

    return model


# ---------------------------
# Recommend Movies
# ---------------------------
def recommend_movies(user_id: int, model: SVD, ratings, movies, movie_titles, n: int = 10):

    all_users = set(ratings["userId"].unique())

    # Cold start handling
    if user_id not in all_users:
        print(f"User {user_id} not found. Showing popular movies instead.")
        popular = (
            ratings.groupby("movieId")["rating"]
            .mean()
            .sort_values(ascending=False)
            .head(n)
        )
        return pd.DataFrame([
            (movie_titles[mid], round(r, 2)) for mid, r in popular.items()
        ], columns=["Movie", "Predicted Rating"])

    seen_movies = set(ratings[ratings["userId"] == user_id]["movieId"])
    all_movies = set(movies["movieId"])

    unseen_movies = list(all_movies - seen_movies)

    predictions = []
    for movie_id in unseen_movies:
        pred = model.predict(user_id, movie_id).est
        predictions.append((movie_id, pred))

    # Sort predictions
    predictions.sort(key=lambda x: x[1], reverse=True)

    top_n = predictions[:n]

    results = [
        (movie_titles.get(movie_id, "Unknown"), round(rating, 2))
        for movie_id, rating in top_n
    ]

    return pd.DataFrame(results, columns=["Movie", "Predicted Rating"])


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    ratings, movies, movie_titles = load_data()

    model = train_model(ratings, retrain=False)

    user_id = 1

    print(f"\nTop recommendations for User {user_id}:\n")
    print(recommend_movies(user_id, model, ratings, movies, movie_titles))