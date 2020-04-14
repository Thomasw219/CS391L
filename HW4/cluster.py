import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import OPTICS

STEP = 0.05
T_I = 0
T_F = 17.5
INIT_SIG_F = 2.33219503
INIT_SIG_L = -4.55283543
INIT_SIG_N = -1.59004312
INTERVAL = 2
time_centers = np.arange(T_I, T_F, STEP)

means = np.load("./means.npy")
sig_f = np.load("./sig_f.npy")
sig_l = np.load("./sig_l.npy")
sig_n = np.load("./sig_n.npy")

data = np.zeros((sig_f.shape[0], 3))
data[:, 0] = sig_f
data[:, 1] = sig_l
data[:, 2] = sig_n

XI = 0.01
"""
for i in range(1, 100):
    model = OPTICS(min_samples=i, xi=XI)
    model.fit(data)
    num_neg = 0
    for x in model.labels_:
        if x < 0:
            num_neg += 1
    print(i, num_neg)
"""
model = OPTICS(min_samples=16, xi=XI)
model.fit(data)

clusters = {}
labels = model.labels_
for i, t in enumerate(time_centers):
    l = labels[i]
    if l not in clusters.keys():
        clusters[l] = ([], [])
    clusters[l][0].append(t)
    clusters[l][1].append(means[i])
print(clusters.keys())

colors = ['r', 'b', 'g', 'm', 'c', 'y', 'k']

plt.figure(0)
for i, k in enumerate(clusters.keys()):
    x = clusters[k][0]
    y = clusters[k][1]
    s = str(k)
    if k == -1:
        s = 'Unclustered/Noise'
    plt.scatter(x, y, s=0.75, c=colors[i], label=s)
plt.legend()
plt.savefig("figures/graphs/clusters.png")

print("Done")
