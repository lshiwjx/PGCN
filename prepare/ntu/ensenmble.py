import pickle
import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['../../'])
import numpy as np
import os

# penna
class_num = 60
right_num = [0 for i in range(class_num)]
total_num = [0 for i in range(class_num)]

label = open('./train_val_test/val/ntu/val_pose_label.pkl', 'rb')
label = np.array(pickle.load(label))

res_pgcn = dict(pickle.load(open('./train_val_test/val/ntu/pgcn_score.pkl', 'rb')).items())
res_i3d = dict(pickle.load(open('./train_val_test/val/ntu/i3d_score.pkl', 'rb')).items())
res_skeleton = list(pickle.load(open('./train_val_test/val/ntu/ske_score.pkl', 'rb')).items())
res_skeleton_bone = list(pickle.load(open('./train_val_test/val/ntu/skebone_score.pkl', 'rb')).items())
for i in range(len(label[0])):
    _, l = label[:, i]
    _, r4 = res_skeleton_bone[i]
    name, r3 = res_skeleton[i]
    name = name[:-9]
    r2 = res_i3d[name]
    r1 = res_pgcn[name]
    r = r1 + r2 + r3 + r4
    r = np.argmax(r)
    l = int(l)
    right_num[l] += int(r == l)
    total_num[l] += 1
print(sum(right_num) / sum(total_num))
