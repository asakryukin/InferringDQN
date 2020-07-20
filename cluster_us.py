import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans

game = "Walker2d-v2"

num_joints = 6
start_from = 1000
end_at = 5000
window = 100

N_TRIALS = 2

struct = "0-1-2-3-4-5.pkl"

joint_labels = ['Hip','Knee','Ankle','LHip','LKnee','LAnkle']

type = "sig"
colors = ['blue', 'red', 'green', 'orange', 'purple', 'black', 'brown', 'pink', 'grey', 'yellow']
line_type = ['-', '--']

fig, axes = plt.subplots(1)
all_trials = np.zeros((num_joints,end_at-start_from))
combined_uncertanitites = np.zeros((num_joints, end_at - start_from))
array_for_std = np.empty((num_joints,0)).tolist()
for trial in range(N_TRIALS):
    qvals = pk.load(open(str(trial)+'q_stat_w'+str(type)+'__noisy_'+game+struct,'rb'))
    game_scores = np.transpose(np.array(qvals))

    alls = np.zeros((2,num_joints))

    for layer in range(2):
        all_scores = []
        for i in range(num_joints):
            t = []
            for j in range(len(game_scores[layer,i])-window):

                t.append(np.mean(game_scores[layer, i][j:j + window]))

            all_scores.append(t)
            if layer == 0:
                combined_uncertanitites[i] += (256.0*128.0)*np.array(t[start_from:end_at])
                array_for_std[i].append((256.0*128.0)*np.array(t[start_from:end_at]))
            elif layer == 1:
                combined_uncertanitites[i] += (128.0*17.0)*np.array(t[start_from:end_at])
                array_for_std[i][-1] += (128.0*17.0)*np.array(t[start_from:end_at])
            alls[layer,i] += np.mean(t[start_from:end_at])

alls = alls/float(N_TRIALS)
array_for_std = np.std(array_for_std, 1, keepdims=False)
legends = ['LHip', 'LKnee', 'LAnkle', 'RHip', 'RKnee', 'RAnkle']
# legends = ['abdomen_y',
# 'abdomen_z',
# 'abdomen_x',
# 'right_hip_x',
# 'right_hip_z',
# 'right_hip_y',
# 'right_knee',
# 'left_hip_x',
# 'left_hip_z',
# 'left_hip_y',
# 'left_knee',
# 'right_shoulder1',
# 'right_shoulder2',
# 'right_elbow',
# 'left_shoulder1',
# 'left_shoulder2',
# 'left_elbow',
#
# ]
for i in range(num_joints):
    combined_uncertanitites[i] = (combined_uncertanitites[i]/(128.0 * 17.0+256.0*128.0))/float(N_TRIALS)
    array_for_std[i] = array_for_std[i]/(128.0 * 17.0+256.0*128.0)/float(N_TRIALS)
    axes.plot(list(range(start_from, end_at)), combined_uncertanitites[i], colors[i % 10],
                           linestyle=line_type[int(i / 10)], label=str(legends[i]))
    axes.fill_between(list(range(start_from, end_at)), combined_uncertanitites[i] - array_for_std[i],
                                   combined_uncertanitites[i] + array_for_std[i], color=colors[i % 10], alpha=0.3)
    axes.set_title(game)
    axes.set_xlabel('Game episode')
    axes.set_ylabel('u$^{\sigma}$')
    axes.legend(loc=2)
plt.show()
distances = []
all_trials = combined_uncertanitites
all_finals = []
for n_c in range(0, num_joints):
    km = TimeSeriesKMeans(n_clusters=n_c + 1, metric="dtw").fit(all_trials)
    joint_means = np.zeros(num_joints)
    labels = km.labels_
    uncerts = np.array(all_trials)
    for joint in range(num_joints):
        joint_means[joint] = np.mean(all_trials[joint])
    cluster_means = np.zeros(n_c + 1)
    for cl_i in range(n_c + 1):
        counter = 0
        for li, l in enumerate(labels):
            if l == cl_i:
                counter += 1
                cluster_means[cl_i] += joint_means[li]
        cluster_means[cl_i] = cluster_means[cl_i] / float(counter)
    cluster_order = list(reversed(np.argsort(cluster_means)))
    final_order = []
    for cl_i in cluster_order:
        final_order.append([])
        for li, l in enumerate(labels):
            if l == cl_i:
                final_order[-1].append(li)
    print('Final order for ' + str(n_c + 1) + ' clusters:')
    print(final_order)
    all_finals.append(final_order)
    distances.append(km.inertia_)
    print(labels)
    print(distances)
distances.append(0.0)
plt.plot(distances)
plt.show()