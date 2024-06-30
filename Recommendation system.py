import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from surprise.accuracy import rmse

# Load the MovieLens dataset using Surprise library
file_path = "/Applications/My Desktop/Internships/ratings.csv"
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file(file_path, reader=reader)

# Split the dataset into a training set and a test set
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train the model
model = SVD()
model.fit(trainset)

predictions = model.test(testset)
accuracy = rmse(predictions)
print("RMSE on test set:", accuracy)

# User_id for whom we want to recommend movies
user_id = str(1)  

# Predict the ratings for the user
movies_to_recommend = []
for movie_id in range(1, 10):  
    
    # Example: recommend top 10 movies
    pred = model.predict(user_id, str(movie_id))
    movies_to_recommend.append(pred)

# Print the recommended movies
print(f"The following movies are recommended for user {user_id}:")
for movie in movies_to_recommend:
    print(f"Movie ID: {movie.iid}, Rating Prediction: {movie.est}")
