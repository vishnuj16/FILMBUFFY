import numpy as np
import pandas as pd
import ast
import nltk
import os
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
print(movies.shape)

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

#dropping movies which have an empty overview (there were a few)
movies.dropna(inplace=True)

#the genre section is a list of dictionaries describing the format, we need the genres in a list
def convert(obj):
    genres = []
    for dic in ast.literal_eval(obj):
        genres.append(dic['name'])
    return genres

def convert5(obj):
    genres = []
    counter = 0
    for dic in ast.literal_eval(obj):
        if counter!=5:
            genres.append(dic['name'])
            counter+=1
        else:
            break
    return genres

def fetch_director(obj):
    genres = []
    for dic in ast.literal_eval(obj):
        if dic['job'] == 'Director':
            genres.append(dic['name'])
            break
    return genres

ps = PorterStemmer()
def stem(text):
    l = []
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)



movies['genres'] = movies['genres'].apply(convert)
#print(movies['genres'].head())

movies['keywords'] = movies['keywords'].apply(convert)
#print(movies['keywords'])

movies['cast'] = movies['cast'].apply(convert5)
#print(movies['cast'])

movies['crew'] = movies['crew'].apply(fetch_director)
#print(movies['crew'])

movies['overview'] = movies['overview'].apply(lambda x:x.split())
# print(movies.head())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
# print(movies['genres'].head())
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])
# print(movies.head())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

#print(new_df['tags'][0])
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
# print(new_df.head())

new_df['tags'] = new_df['tags'].apply(stem)

#VECTORIZATION
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
#print(vectors[0])

# print(cv.get_feature_names_out())
similarity = cosine_similarity(vectors)
# print(similarity[0])

#print(new_df[new_df['title'] == 'The Dark Knight'].index[0])

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    count =0
    print("\nHere are some movies specifically Recommended for you !\n")
    for i in movies_list:
        if count!=5:
            print(new_df.iloc[i[0]].title)
            count+=1


os.system("cls")
movie = input("Which is you favourite movie?\n")
recommend(movie)
