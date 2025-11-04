import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocessing():

    # Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    clean_path = os.path.join(project_root, "data/cleaned_spotify_data.csv")
    features_path = os.path.join(project_root, "data/preprocessed_spotify_data.csv")
    
    # Load raw data
    df = pd.read_csv(clean_path)

    # Define feature groups
    numeric_features = [
        'duration_ms', 'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    categorical_features = ['key', 'time_signature', 'track_genre']
    binary_features = ['explicit', 'mode']
    target_feature = ['popularity']

    # Numeric scaling
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(df[numeric_features])

    # Categorical encoding
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cats = encoder.fit_transform(df[categorical_features])
    cat_feature_names = encoder.get_feature_names_out(categorical_features)

    # Combining Binary and target
    binary_and_target = df[binary_features + target_feature].to_numpy()

    # Combine everything
    feature_array = np.hstack([scaled_numeric, encoded_cats, binary_and_target])
    feature_names = numeric_features + list(cat_feature_names) + binary_features + target_feature

    # Create DataFrame
    features_df = pd.DataFrame(feature_array, columns=feature_names)

    # Save preprocessed data
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    features_df.to_csv(features_path, index=False)
    
    print(f"Feature data saved to: {features_path}")
    return features_df