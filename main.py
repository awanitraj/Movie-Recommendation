from src.data_preprocessing import load_and_merge_data
from src.recommender import clean_and_vectorize, recommend

def main():
    movie_path = "data/tmdb_5000_movies.csv"
    credit_path = "data/tmdb_5000_credits.csv"

    movies = load_and_merge_data(movie_path, credit_path)
    
    print("\nFirst 5 rows of the dataset:")
    print(movies.head())
    

    similarity, movies = clean_and_vectorize(movies)
    
    title = input("Enter a movie title to get recommendations: ")
    

    results = recommend(title, similarity, movies)
    
    if results:  
        print("\nTop 5 movie recommendations:")
        for movie in results:
            print(movie)
    else:
        print("No recommendations found.")
        
def recommend(title, similarity, movies):

    matched_movies = movies[movies['title'].str.lower() == title.lower()]

    if matched_movies.empty:
        print(f"❌ Movie titled '{title}' not found in the dataset.")
        print("✅ Here are some available movie titles you can try:")
        print(movies['title'].head(10).to_list()) 
        return []
    
    movie_index = matched_movies.index[0]
    distances = list(enumerate(similarity[movie_index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    
    recommended_titles = [movies.iloc[i[0]].title for i in sorted_movies]
    return recommended_titles

if __name__ == "__main__":
    main()
