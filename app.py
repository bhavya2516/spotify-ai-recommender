from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

app = Flask(__name__)

# Load ML model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load dataset
df = pd.read_csv("SpotifyFeatures.csv")

# Select important audio features
features = df[['danceability', 'energy', 'tempo', 'loudness', 'valence']]

# Scale features
scaled = scaler.transform(features)

# Predict clusters
clusters = model.predict(scaled)
df['Cluster'] = clusters

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)

df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]


# Home Page
@app.route('/')
def home():
    return render_template("index.html")


# Song Recommendation Route
@app.route('/recommend', methods=['POST'])
def recommend():

    song_name = request.form['song']

    # Find song in dataset
    song = df[df['track_name'].str.contains(song_name, case=False, na=False)]

    if song.empty:
        return render_template(
            "index.html",
            prediction="Song not found in dataset"
        )

    # Get cluster of selected song
    cluster = song.iloc[0]['Cluster']

    # Get recommendations from same cluster
    recommendations = df[df['Cluster'] == cluster].sample(10)

    songs = recommendations[['track_name', 'artist_name']].values.tolist()

    return render_template(
        "index.html",
        prediction=f"Songs similar to {song_name}",
        songs=songs
    )


# Visualization Route
@app.route('/visualize')
def visualize():

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df['PCA1'],
        df['PCA2'],
        c=df['Cluster'],
        cmap='viridis'
    )

    plt.title("Spotify Song Clusters")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")

    plt.savefig("static/clusters.png")
    plt.close()

    return render_template("visualize.html")


# Run Flask App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
