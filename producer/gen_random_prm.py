import sys
import numpy as np

np.set_printoptions(threshold='nan')

#fnames = filtered_list_dir('/home/chenzhixing/data/BU3D/disks', ['mat'])
#ndata = len(fnames)

pnum = 60
enum = 12
ldm_num = 68
prm60 = np.random.permutation(pnum)
prm720 = np.zeros((pnum*enum,)).astype(np.int32)
prm = np.zeros((pnum*enum*ldm_num,)).astype(np.int32)
for i in range(len(prm60)):
  prm12 = np.random.permutation(enum)
  prm720[i*enum:(i+1)*enum] = prm12 + prm60[i]*enum
arr68 = np.array([i for i in range(ldm_num)]).astype(np.int32)
for i in range(len(prm720)):
  prm[i*ldm_num:(i+1)*ldm_num] = arr68 + prm720[i]*ldm_num
#prm = np.array([i for i in range(10)]).astype(np.int32)
print(prm)
print(prm.shape)
print(type(prm[0]))
print(np.max(prm))
print(np.min(prm))
prm.tofile("prm.bin")


#ndata = 720
#print(ndata)
#prm = np.random.permutation(ndata)
#print(prm)
#print(prm.shape)
#print(type(prm[0]))
#print(np.max(prm))
#print(np.min(prm))
#prm.tofile("prm.bin")
