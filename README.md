# 🎬 Movie Recommendation System

This is a **Content-Based Movie Recommender System** built using Python. It recommends similar movies based on textual features like genres, keywords, cast, and overview.

---

## 📌 Features

- Recommends top similar movies based on a selected title
- Uses content similarity via **TF-IDF Vectorizer** and **cosine similarity**
- Clean modular code with `src/` folder
- NLTK preprocessing with stopword removal
- Lightweight and fast response

---

## 🗃️ Dataset

The project uses the **TMDB 5000 Movie Dataset** from Kaggle:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

📥 Download here: [TMDB Dataset - Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

Place the CSV files inside the `data/` directory.

---

## 📂 Project Structure


---

## ⚙️ How It Works

1. Loads and merges movie and credits data
2. Cleans and combines features like `overview`, `genres`, `cast`, `keywords`
3. Applies **TF-IDF Vectorization**
4. Computes **cosine similarity**
5. Fetches top N similar movies

---

## ▶️ Run the Project

### 1. Install Required Libraries:

```bash
pip install pandas numpy scikit-learn nltk
