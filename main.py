from src.data_preprocessing import load_and_merge_data
from src.recommender import clean_and_vectorize, recommend

def main():
    movie_path = "data/tmdb_5000_movies.csv"
    credit_path = "data/tmdb_5000_credits.csv"

    movies = load_and_merge_data(movie_path, credit_path)
    similarity, movies = clean_and_vectorize(movies)

    title = input("Enter a movie title to get recommendations: ")
    results = recommend(title, similarity, movies)

    print("\nTop 5 movie recommendations:")
    for movie in results:
        print(movie)

if __name__ == "__main__":
    main()
