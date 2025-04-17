import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_and_vectorize(data):
    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    data['genres'] = data['genres'].apply(convert)
    data['keywords'] = data['keywords'].apply(convert)
    data['cast'] = data['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3])
    data['crew'] = data['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])

    data['tags'] = data['genres'] + data['keywords'] + data['cast'] + data['crew']
    data['tags'] = data['tags'].apply(lambda x: ' '.join(x))
    
    tfidf = TfidfVectorizer(stop_words='english')
    vectors = tfidf.fit_transform(data['tags'])
    
    similarity = cosine_similarity(vectors)
    return data, similarity
