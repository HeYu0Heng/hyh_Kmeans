import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Microsoft YaHei')
img = np.array(Image.open('color.jpg'))
hang, lie, dims = img.shape
print(hang)
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]
list_R, list_G, list_B = [], [], []
reshaped_R = R.reshape(1, -1)
reshaped_G = G.reshape(1, -1)
reshaped_B = B.reshape(1, -1)
print(np.size(reshaped_B))
A = np.vstack((reshaped_R, reshaped_G, reshaped_B))
kmeans = KMeans(n_clusters=3,init='random',n_init=10,tol=0.001)
kmeans.fit(A.T)
labels = kmeans.labels_
# 聚类结果
center = kmeans.cluster_centers_
# 聚类中心
print(center)
B = np.zeros_like(img)
for i in range(hang):
    for j in range(lie):
        label = labels[i * lie + j]
        B[i, j, 0] = center[label, 0]
        B[i, j, 1] = center[label, 1]
        B[i, j, 2] = center[label, 2]
        print(B)
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(B)
plt.title('Segmented Image')

plt.show()