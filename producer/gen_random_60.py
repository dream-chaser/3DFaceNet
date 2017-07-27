import numpy as np

np.set_printoptions(threshold='nan')

num = [56, 44]
valid = [[],[]]

f = open('3_4_error.txt','r')
for i,line in enumerate(f):
  line = line.strip()
  line = list(map(eval,line.split(' ')[1:]))
  for j in xrange(num[i]):
    if j+1 not in line:
      valid[i].append(j+1)
f.close()

prm_f = np.random.permutation(valid[0])[:30]
prm_m = np.random.permutation(valid[1])[:30]
for i,m in enumerate(prm_m):
  prm_m[i] = m + num[0]
prm_f = list(prm_f)
prm_m = list(prm_m)
prm = np.array(prm_f + prm_m)
prm = np.random.permutation(prm)
prm.tofile("use_data_60.bin")
