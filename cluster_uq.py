import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans

colors = ['blue', 'red', 'green', 'orange', 'purple', 'black','brown','pink','grey','yellow']
line_type = ['-','--']

prefix = 'q_stat_stds99__Hopper-v2_0-1-2.pkl'
log_indexes = [0,1,2,3,4,5]

y_label = "$u^\sigma$"

start_ind = 0
end_index = 3500

num_joints = 3

colors = ['blue', 'red', 'green', 'orange', 'purple', 'black','brown','pink','grey','yellow']
line_type = ['-','--']

fig, axes = plt.subplots(1,1)

N_TRIALS = 3

all_data = np.empty((N_TRIALS,0)).tolist()

all_means = []
all_stds = []


all_scores = []

all_ts = []

np.random.shuffle(log_indexes)
indexes = log_indexes
print(indexes)
indexes = indexes[:N_TRIALS]
running_window_size = 100
for i, ir in enumerate(indexes):
    uncerts = np.array(pkl.load(open(str(ir)+prefix, 'rb')))
    uncerts = np.array(uncerts)[start_ind:end_index]
    curr_ts = []
    for u_i in range(uncerts.shape[1]):
        t = []
        for j in range(uncerts.shape[0] -running_window_size):
            t.append(np.mean(uncerts[j:j + running_window_size,u_i]))
        curr_ts.append(t)
    all_data[i]=np.array(curr_ts)
    all_ts.append(curr_ts)
all_data = np.array(all_data)
uncerts = np.mean(np.array(all_data),0)
sts = np.std(np.array(all_data),0)
all_means.append(uncerts)
all_stds.append(sts)

distances = []
all_finals = []
for n_c in range(num_joints):
    km = TimeSeriesKMeans(n_clusters=n_c + 1, metric="dtw").fit(uncerts)
    joint_means = np.zeros(num_joints)
    labels = km.labels_
    uncerts = np.array(uncerts)
    for joint in range(num_joints):
        joint_means[joint] = np.mean(uncerts[joint])
    cluster_means = np.zeros(n_c+1)
    for cl_i in range(n_c+1):
        counter = 0
        for li,l in enumerate(labels):
            if l == cl_i:
                counter+=1
                cluster_means[cl_i] += joint_means[li]
        cluster_means[cl_i] = cluster_means[cl_i]/float(counter)
    cluster_order = list(reversed(np.argsort(cluster_means)))
    final_order = []
    for cl_i in cluster_order:
        final_order.append([])
        for li, l in enumerate(labels):
            if l == cl_i:
                final_order[-1].append(li)
    print('Final order for '+str(n_c+1)+' clusters:')
    print(final_order)
    all_finals.append(final_order)
    distances.append(km.inertia_ )
    print(labels)
    print(distances)


print("--------------------------------------------------")
plt.plot(distances)
plt.show()
