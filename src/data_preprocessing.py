import pandas as pd
import ast

def load_and_merge_data(movie_path, credit_path):
    movies = pd.read_csv(movie_path)
    credits = pd.read_csv(credit_path)

    movies = movies.merge(credits, on='title')

    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

    for col in ['genres', 'keywords', 'cast', 'crew']:
        movies[col] = movies[col].apply(ast.literal_eval)

    return movies
