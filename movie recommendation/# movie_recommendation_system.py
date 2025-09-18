# movie_recommendation_system.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Loads movie and rating data from CSV files."""
    try:
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
        print("✅ Data loaded successfully!")
        return movies, ratings
    except FileNotFoundError:
        print("❌ Error: Make sure 'movies.csv' and 'ratings.csv' are in the same directory.")
        return None, None

def create_user_item_matrix(ratings, movies):
    """Merges ratings and movies data and creates a user-item matrix."""
    df = pd.merge(ratings, movies, on='movieId')
    
    # Create a user-item matrix: rows are users, columns are movie titles, values are ratings.
    # We use pivot_table to handle this transformation.
    # fill_value=0 means if a user hasn't rated a movie, their rating is 0.
    user_item_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    
    print("✅ User-item matrix created.")
    return user_item_matrix, df

def get_recommendations(user_id, user_item_matrix, movies_df):
    """
    Generates movie recommendations for a given user using cosine similarity.
    """
    if user_id not in user_item_matrix.index:
        print(f"❌ Error: User ID {user_id} not found.")
        return

    # 1. Calculate Cosine Similarity
    # This measures the similarity between our target user and all other users.
    # The result is a matrix where each value is the similarity score between two users.
    user_similarities = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarities, index=user_item_matrix.index, columns=user_item_matrix.index)

    # 2. Find Similar Users
    # Get the similarity scores for our target user and sort them.
    # We drop the user's own similarity score (which is 1.0) and take the top 10 most similar users.
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id).head(10)
    
    if similar_users.empty:
        print("Could not find any similar users.")
        return

    print(f"\nTop similar users for User ID {user_id}:\n{similar_users}\n")

    # 3. Find Movies to Recommend
    # Get the movies rated by these similar users.
    similar_users_ratings = user_item_matrix.loc[similar_users.index]

    # Find movies that our target user has NOT rated (rating == 0).
    unseen_movies_mask = user_item_matrix.loc[user_id] == 0
    unseen_movies = user_item_matrix.columns[unseen_movies_mask]

    # We only want to recommend movies that the target user hasn't seen yet.
    recommendable_movies = similar_users_ratings[unseen_movies]

    # Calculate the weighted average score for each movie based on user similarity.
    # Movies rated highly by very similar users will get a higher score.
    recommendation_scores = recommendable_movies.mean(axis=0).sort_values(ascending=False)

    # 4. Return Top N Recommendations
    # Add genre information to the final recommendations.
    top_10_recommendations = recommendation_scores.head(10)
    recommended_movies_details = movies_df[movies_df['title'].isin(top_10_recommendations.index)]
    
    print(f"--- Top 10 Movie Recommendations for User ID {user_id} ---")
    for title, score in top_10_recommendations.items():
        # A simple way to display details, can be improved.
        details = recommended_movies_details[recommended_movies_details['title'] == title].iloc[0]
        print(f"🎬 Title: {details['title']}, Genre: {details['genres']} (Score: {score:.2f})")


def main():
    """Main function to run the recommendation system."""
    movies, ratings = load_data()
    if movies is None or ratings is None:
        return

    user_item_matrix, merged_df = create_user_item_matrix(ratings, movies)
    
    while True:
        try:
            user_id_input = input("\nEnter a User ID to get recommendations for (e.g., 1 to 610), or 'exit' to quit: ")
            if user_id_input.lower() == 'exit':
                break
            target_user_id = int(user_id_input)
            get_recommendations(target_user_id, user_item_matrix, merged_df)
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()