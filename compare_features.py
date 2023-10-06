# pet-celebrity-lookalike-backend/compare_features.py
import json
from scipy.spatial.distance import cosine


def load_celebrity_features(celebrity_features_path):
    with open(celebrity_features_path, 'r') as f:
        celebrity_features = json.load(f)
    return celebrity_features


def calculate_similarity(feature1, feature2):
    # Using cosine similarity, but need to try other distance metrics
    return 1 - cosine(feature1, feature2)


def find_closest_matches(user_features, celebrity_features, num_matches=3):
    # Calculate similarity with each celebrity
    similarities = [
        {
            'label': celeb['label'],
            'similarity': calculate_similarity(user_features, celeb['features'])
        }
        for celeb in celebrity_features
    ]

    # Sort by similarity (higher is better)
    sorted_matches = sorted(similarities, key=lambda x: x['similarity'], reverse=True)

    # Return the top N matches
    return sorted_matches[:num_matches]


if __name__ == "__main__":
    celebrity_features_path = 'celebrity_features.json'
    user_features_path = 'user_features.json'
    num_matches = 3

    celebrity_features = load_celebrity_features(celebrity_features_path)

    # Load user features
    with open(user_features_path, 'r') as f:
        user_features = json.load(f)

    # Find closest matches
    closest_matches = find_closest_matches(user_features, celebrity_features, num_matches)

    print("Closest Matches:")
    for match in closest_matches:
        print(f"Label: {match['label']}, Similarity: {match['similarity']}")
