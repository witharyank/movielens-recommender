import pandas as pd
import joblib
import os
from surprise import Dataset, Reader, SVD

ratings = pd.read_csv("../data/ratings.csv")
movies = pd.read_csv("../data/movies.csv")

movie_titles = dict(zip(movies.movieId, movies.title))

MODEL_PATH = "svd_model.pkl"


def train_model(random_state: int = 42) -> SVD:

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

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


def recommend_movies(user_id: int, model: SVD, n: int = 10):

    if user_id not in ratings["userId"].unique():
        raise ValueError(f"User {user_id} not found")

    seen_movies = set(ratings[ratings["userId"] == user_id]["movieId"])
    all_movies = set(movies["movieId"])
    unseen_movies = list(all_movies - seen_movies)

    predictions = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in unseen_movies
    ]

    predictions.sort(key=lambda x: x[1], reverse=True)

    top_n = predictions[:n]

    results = [
        (movie_titles[movie_id], round(rating, 2))
        for movie_id, rating in top_n
    ]

    return pd.DataFrame(results, columns=["Movie", "Predicted Rating"])


if __name__ == "__main__":

    model = train_model()
    user_id = 1

    print(f"Top recommendations for User {user_id}")
    print(recommend_movies(user_id, model))