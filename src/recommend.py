import pandas as pd
from surprise import Dataset, Reader, SVD


# Load data
ratings = pd.read_csv("../data/ratings.csv")
movies = pd.read_csv("../data/movies.csv")


def train_model(random_state: int = 42) -> SVD:
    """
    Train and return an SVD recommendation model.
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings[["userId", "movieId", "rating"]],
        reader
    )

    trainset = data.build_full_trainset()
    model = SVD(random_state=random_state)
    model.fit(trainset)
    return model


def recommend_movies(user_id: int, model: SVD, n: int = 10):
    """
    Recommend top-N movies for a given user.
    """
    if user_id not in ratings["userId"].values:
        raise ValueError(f"User {user_id} not found in ratings data")

    seen_movies = set(ratings[ratings["userId"] == user_id]["movieId"])
    all_movies = set(movies["movieId"])
    unseen_movies = all_movies - seen_movies

    predictions = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in unseen_movies
    ]

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n = predictions[:n]

    results = []
    for movie_id, rating in top_n:
        title = movies.loc[movies["movieId"] == movie_id, "title"].values[0]
        results.append((title, round(rating, 2)))

    return results


if __name__ == "__main__":
    model = train_model()
    user_id = 1

    print(f"Top recommendations for User {user_id}:")
    for title, rating in recommend_movies(user_id, model):
        print(f"{title} (predicted rating: {rating})")
