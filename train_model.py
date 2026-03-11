import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

df = pd.read_csv("SpotifyFeatures.csv")

features = df[['danceability','energy','tempo','loudness','valence']]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)

model = KMeans(n_clusters=5)
model.fit(scaled)

pickle.dump(model, open("model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))

print("Model Saved Successfully")