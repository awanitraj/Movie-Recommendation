from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

def clean_and_vectorize(movies):
    def convert(obj):
        return [i['name'] for i in obj][:3]

    def get_director(obj):
        for i in obj:
            if i['job'] == 'Director':
                return [i['name']]
        return []

    def collapse(lst):
        return ' '.join(i.replace(" ", "") for i in lst)

    movies['tags'] = movies['overview'].fillna('') + ' ' + \
        movies['genres'].apply(convert).apply(collapse) + ' ' + \
        movies['keywords'].apply(convert).apply(collapse) + ' ' + \
        movies['cast'].apply(convert).apply(collapse) + ' ' + \
        movies['crew'].apply(get_director).apply(collapse)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)
    return similarity, movies

def recommend(title, similarity, movies):
    movie_index = movies[movies['title'] == title].index[0]
    distances = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in distances:
        recommendations.append(movies.iloc[i[0]].title)
    return recommendations
