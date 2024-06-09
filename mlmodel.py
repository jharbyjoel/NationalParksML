from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

national_parks_csv = pd.read_csv('parks.csv')

X = national_parks_csv[['Latitude', 'Longitude']]

number_of_clusters = 5

kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(X)

national_parks_csv['Cluster'] = kmeans.labels_

plt.figure(figsize=(12,8))

for cluster in range(number_of_clusters):
    cluster_parks = national_parks_csv[national_parks_csv['Cluster'] == cluster]
    plt.scatter(cluster_parks['Longitude'], cluster_parks['Latitude'], label=f'Cluster {cluster + 1}')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('National Parks Clustering')
plt.legend()
plt.show()
