from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

app = Flask(__name__)

# load model
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

# load dataset
df = pd.read_csv("SpotifyFeatures.csv")

features = df[['danceability','energy','tempo','loudness','valence']]

scaled = scaler.transform(features)

clusters = model.predict(scaled)
df['Cluster'] = clusters


# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)

df['PCA1'] = pca_result[:,0]
df['PCA2'] = pca_result[:,1]


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/recommend', methods=['POST'])
def recommend():

    song_name = request.form['song']

    song = df[df['track_name'].str.contains(song_name, case=False, na=False)]

    if song.empty:
        return render_template("index.html",
                               prediction="Song not found in dataset")

    cluster = song.iloc[0]['Cluster']

    recommendations = df[df['Cluster']==cluster].head(10)

    songs = recommendations[['track_name','artist_name']].values.tolist()

    return render_template("index.html",
                           prediction=f"Songs similar to {song_name}",
                           songs=songs)
@app.route('/visualize')
def visualize():

    plt.figure(figsize=(8,6))

    plt.scatter(df['PCA1'], df['PCA2'], c=df['Cluster'], cmap='viridis')

    plt.title("Spotify Song Clusters")

    plt.xlabel("PCA1")
    plt.ylabel("PCA2")

    plt.savefig("static/clusters.png")

    return render_template("visualize.html")


if __name__ == "__main__":
    app.run(debug=True)