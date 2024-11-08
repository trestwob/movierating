import pandas as pd
movies = pd.read_csv('ml-latest-small/movies.csv', delimiter=',', header=None, names=['movieId', 'title', 'genres']).drop(0)
ratings = pd.read_csv('ml-latest-small/ratings.csv', delimiter=',', header=None, names=['userId', 'movieId', 'rating', 'timestamp']).drop(0)

movie_data = pd.merge(ratings, movies[['movieId', 'title']])