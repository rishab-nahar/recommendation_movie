#creating a movie recomendor system(content based)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

movies_D = pd.read_csv("movie_metadata.csv", error_bad_lines=False)
features = [1, 9, 10, 11, 16, 21, 19]
feature_set = movies_D.iloc[:, features].values
#importing the required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cs
import re
from sklearn.impute import SimpleImputer

missingvalues = SimpleImputer(missing_values=np.nan, strategy='most_frequent', verbose=1)
feature_set = missingvalues.fit_transform(feature_set)
#creating the bag of words
for i in range(5043):
    feature_set[i][1] = " ".join(feature_set[i][1].strip().split("|"))
    feature_set[i][4] = " ".join(feature_set[i][4].strip().split("|"))


def join_features(row):
    return row[0] + " " + row[1] + " " + row[2] + " " + row[3] + " " + row[4] + " " + row[5]


joined_features = []
for i in range(5043):
    joined_features.append(join_features(feature_set[i]))

cv = CountVectorizer()
count_m = cv.fit_transform(joined_features)
similar = cs(count_m)

print("ENTER THE MOVIE YOU LIKE")
liked = input()
index = -1
for i in range(5043):
    if re.sub("\s*", "", feature_set[i][3]).lower() == re.sub("\s*", "", liked).lower():
        index = i
        break
if index == -1:
    print("{}movie does not exist".format(liked))
else:
    similar_movies = sorted(enumerate(similar[index]), key=lambda x: x[1], reverse=True)
    print("recommending 10 movies")
    for i in range(1, 11):
        print(str(i) + ". " + feature_set[similar_movies[i][0]][3], end="\t")

