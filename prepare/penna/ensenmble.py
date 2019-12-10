import pickle
import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['../../'])
import numpy as np
import os

# penna
class_num = 15
right_num = [0 for i in range(class_num)]
total_num = [0 for i in range(class_num)]

label = open('./train_val_test/val/penna/val_pose_label.pkl', 'rb')
label = np.array(pickle.load(label))

res_pgcn = dict(list(pickle.load(open('./train_val_test/val/penna/pgcn_score.pkl', 'rb')).items()))
res_i3d = dict(list(pickle.load(open('./train_val_test/val/penna/i3d_score.pkl', 'rb')).items()))
res_skeleton = dict(list(pickle.load(open('./train_val_test/val/penna/ske_score.pkl', 'rb')).items()))
for i in range(len(label[0])):
    name, l = label[:, i]
    r1 = res_pgcn[name]
    r2 = res_i3d[name]
    r3 = res_skeleton[name]
    r = r1 * 2 + r2 + r3
    r = np.argmax(r)
    l = int(l)
    right_num[l] += int(r == l)
    total_num[l] += 1
print(sum(right_num) / sum(total_num))
