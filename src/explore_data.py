import pandas as pd
import matplotlib.pyplot as plt

# Load data
ratings = pd.read_csv("../data/ratings.csv")
movies = pd.read_csv("../data/movies.csv")

print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape)

print("\nRatings head:")
print(ratings.head())

print("\nMovies head:")
print(movies.head())

# Rating distribution
plt.hist(ratings["rating"], bins=10)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()
