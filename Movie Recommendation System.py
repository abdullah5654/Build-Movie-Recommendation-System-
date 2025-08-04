"""
Movie Recommendation System using Content-Based Filtering with Loop Input

- Uses cosine similarity to recommend similar movies based on genres.
- Runs in a loop to accept multiple user inputs.
- Exits when user types 'quit' or 'exit'.

Libraries:
- pandas
- sklearn
- seaborn, matplotlib (optional for visualization)
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Create sample movie dataset with genres
movies_data = {
    'movie_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'title': [
        'The Dark Knight', 'Inception', 'Interstellar', 'Avengers: Endgame',
        'Titanic', 'The Notebook', 'The Matrix', 'Joker'
    ],
    'genres': [
        ['Action', 'Drama', 'Crime'],
        ['Action', 'Sci-Fi', 'Thriller'],
        ['Adventure', 'Drama', 'Sci-Fi'],
        ['Action', 'Adventure', 'Sci-Fi'],
        ['Romance', 'Drama'],
        ['Romance', 'Drama'],
        ['Sci-Fi', 'Action'],
        ['Crime', 'Drama', 'Thriller']
    ]
}

# Step 2: Create DataFrame
df_movies = pd.DataFrame(movies_data)

# Step 3: Convert genres to binary vectors (One-hot encoding)
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df_movies['genres'])
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)

# Step 4: Compute cosine similarity between movies based on genres
similarity_matrix = cosine_similarity(genre_df)

# Optional: Visualize similarity matrix
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, xticklabels=df_movies['title'], yticklabels=df_movies['title'], cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Cosine Similarity between Movies (by Genre)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Step 5: Function to recommend similar movies
def recommend_movies(movie_title, top_n=3):
    """
    Recommends top N similar movies based on genre similarity
    """
    if movie_title not in df_movies['title'].values:
        print("❌ Movie not found in dataset. Please try another title.")
        return
    
    idx = df_movies.index[df_movies['title'] == movie_title][0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the movie itself
    top_movies = [df_movies['title'][i] for i, score in similarity_scores[1:top_n+1]]
    
    print(f"\n🎬 Recommendations for '{movie_title}':")
    for movie in top_movies:
        print(f"👉 {movie}")

# --- Loop-based CLI input
print("🎥 Welcome to the Movie Recommender!")
print("Type a movie name to get recommendations.")
print("Type 'quit' or 'exit' to end the program.\n")

while True:
    user_input = input("Enter movie title: ").strip()
    if user_input.lower() in ['quit', 'exit']:
        print("👋 Exiting the recommender. Goodbye!")
        break
    recommend_movies(user_input)
    print()  # for spacing
