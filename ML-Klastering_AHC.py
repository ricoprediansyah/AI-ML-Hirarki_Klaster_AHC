# ML-6 Hierarchical Clustering: Agglomerative Hierarchical Clustering (AHC)

# Import Library
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Load Dataset
data = pd.read_csv('Mall_Customers (1).csv', index_col='CustomerID')
data

# memeriksa kelengkapan dataset.
data.info()

# Kita hanya memilih dua variable saja, yakni AnnualIncome dan SpendingScore.
# Note: Jika kita memilih variable Gender, maka kita perlu melakukan preprocessing terlebih dahulu.
X = data[['AnnualIncome', 'SpendingScore']].values
X

# Karena standar besar/kecil di kedua variabel berbeda, maka kita perlu melakukan standarisasi terlebih dahulu menggunakan StandardScaler.
scaler = StandardScaler()
scaler = scaler.fit(X)
X = scaler.transform(X)
X

# Membentuk Dendogram
# Pilih metode complete linkage!

# FYI, metode complete linkage bisa diubah dengan:
# . single
# . average
dendrogram = sch.dendrogram(sch.linkage(X, method='complete'))

# Terlihat bahwa garis cluster terpanjang berada dijumlah cluster 2, 3, 4, atau 5.

# Modeling
# Kita akan coba membuat model menggunakan algoritma Agglomerative Hierarchical Clustering (AHC) dengan:

# 4 cluster
# . persamaan jarak menggunakan Euclidean Distance
# . menggunakan metode complete linkage
# Hyperparameter tuning
jumlah_cluster = 4
persamaan_jarak = 'euclidean'
metode_linkage = 'complete'

model_AHC = AgglomerativeClustering(n_clusters=jumlah_cluster, affinity=persamaan_jarak, linkage=metode_linkage)
model_AHC.fit(X)

# Hasil Clustering
labels_agglo = model_AHC.labels_
labels_agglo

# Visualisasi Hasil Clustering
colors = ['red', 'green','blue', 'purple', 'magenta', 'orange', 'yellow']

for i in range(jumlah_cluster):
  plt.scatter(X[labels_agglo==i, 0], X[labels_agglo==i, 1], s=50, marker='o', color=colors[i])

# Kita bisa lihat hasil clusteringnya sudah bagus, terlihat antar cluster
# cukup terpisah (tidak ada misalnya anggota cluster warna biru berada di tengah2 warna hijau).

# Artinya kita bisa menggunakan hasil clustering ini.
data['Hasil_Clustering'] = labels_agglo
data

# Kesimpulan
# Contoh:

# Mencari rata-rata tiap cluster untuk menarik kesimpulan.
for i in range(jumlah_cluster):
  print(f'Cluster ke-{i}')
  print('Rata-rata pemasukan customer   : ', data[data['Hasil_Clustering']==i]['AnnualIncome'].mean())
  print('Rata-rata pengeluaran customer : ', data[data['Hasil_Clustering']==i]['SpendingScore'].mean())
  print()
#   Bisa kita simpulkan bahwa :

# Cluster 0 adalah kelompok customer sangat kaya dan sangat hemat (pengeluarannya sangat kecil)
# Cluster 1 adalah kelompok customer menengah ke bawah dan boros (pengeluaran lebih besar dari pemasukan)
# Cluster 3 adalah kelompok customer menengah ke atas dan boros (pengeluaran lebih besar dari pemasukan)
# Cluster 1 adalah kelompok customer menengah ke bawah dan hemat (pengeluaran lebih kecil dari pemasukan)
