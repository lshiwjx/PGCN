import pickle
import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['../../'])
import numpy as np
import os

# jmdb
ratios = [[1, 1, 1], [3, 1, 1], [1, 1, 2]]
acc = []
class_num = 12
for i in range(1, 4):
    right_num = [0 for i in range(class_num)]
    total_num = [0 for i in range(class_num)]
    label = open('./train_val_test/val/subjhmdb/val_pose_labelsub{}.pkl'.format(i), 'rb')
    label = np.array(pickle.load(label))
    res_pgcn = dict(list(pickle.load(open('./train_val_test/val/subjhmdb/pgcn_score_s{}.pkl'.format(i), 'rb')).items()))
    res_i3d = dict(list(pickle.load(open('./train_val_test/val/subjhmdb/i3d_score_s{}.pkl'.format(i), 'rb')).items()))
    res_skeleton = dict(list(pickle.load(open('./train_val_test/val/subjhmdb/ske_score_s{}.pkl'.format(i), 'rb')).items()))
    for j in range(len(label[0])):
        name, l = label[:, j]
        r1 = res_pgcn[name]
        r2 = res_i3d[name]
        r3 = res_skeleton[name]
        r = r1 * ratios[i-1][0] + r2 * ratios[i-1][1] + r3 * ratios[i-1][2]
        r = np.argmax(r)
        l = int(l)
        right_num[l] += int(r == l)
        total_num[l] += 1
    acc.append(sum(right_num) / sum(total_num))
print(sum(acc) / 3)
