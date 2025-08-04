# Build-Movie-Recommendation-System-
Movie Recommendation System using Content-Based Filtering (Genre-Based Similarity)

This script implements a basic movie recommender system using content-based filtering.
It uses cosine similarity to compare movies based on their genres.

Features:
- A manually created small movie dataset with genres
- Cosine similarity using sklearn to compute similarities
- Genre data encoded using MultiLabelBinarizer
- Recommends top 3 similar movies when user inputs a movie title
- Runs in a loop until the user types 'quit' or 'exit'
- Displays recommendations and handles unknown inputs
- Optional heatmap to visualize similarity matrix

Requirements:
- pandas
- sklearn
- seaborn and matplotlib (for visualization)

Example:
Enter movie title: Inception  
🎬 Recommendations for 'Inception':  
👉 Avengers: Endgame  
👉 The Matrix  
👉 Interstellar
