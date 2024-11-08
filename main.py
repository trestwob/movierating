import pandas as pd
from surprise import SVD, Reader, Dataset
from surprise import accuracy   
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

movies = pd.read_csv('ml-latest-small/movies.csv', delimiter=',', header=None, names=['movieId', 'title', 'genres']).drop(0)
ratings = pd.read_csv('ml-latest-small/ratings.csv', delimiter=',', header=None, names=['userId', 'movieId', 'rating', 'timestamp']).drop(0)

movie_data = pd.merge(ratings, movies[['movieId', 'title']])

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

svd = SVD()
svd.fit(trainset)

predictions = svd.test(testset)
rmse = accuracy.rmse(predictions)
print(f"Rmse: {rmse}")

##movie rating prediction
userId = 1
moviesPredict = ratings[ratings['userId'] == userId]['movieId'].tolist()
total_movies = set(ratings['movieId'].unique()) - set(moviesPredict)
predictions = [svd.predict(userId, movieId) for movieId in total_movies]
predictions.sort(key=lambda x: x.est, reverse=True)
top5rec = predictions[:5]
rec_movie_titles = [movies[movies['movieId'] == pred.iid]['title'].values[0] for pred in top5rec]
print("Top 5 movie -> ")
for title in rec_movie_titles:
    print(title)

plt.figure(figsize=(8, 6))
sns.histplot(ratings['rating'], bins=5, kde=False)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()