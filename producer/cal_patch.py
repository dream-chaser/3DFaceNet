import theano
import theano.tensor as T
import theano.sparse as Tsp
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import math

ntheta = 8
nrho = 2
max_rho = 50
#sigma = 10
offset = 1000000

t0 = np.zeros(nrho,dtype=theano.config.floatX)
r0 = np.zeros(ntheta,dtype=theano.config.floatX)
theta_v = np.array([(tv*2+1)*math.pi/ntheta for tv in range(ntheta)],dtype=theano.config.floatX).reshape(ntheta,1) + t0
rho_v = np.array([(rv+0.5)*max_rho/nrho for rv in range(nrho)],dtype=theano.config.floatX).reshape(nrho,1) + r0

theta = theano.shared(value=theta_v.transpose().flatten(), name='theta_op', borrow=True)
rho = theano.shared(value=rho_v.flatten(), name='rho_op', borrow=True)

x_local = Tsp.csr_matrix('x_local')
y_local = Tsp.csr_matrix('y_local')

def cal_patch(theta_i, rho_i, x_coord, y_coord):
  one_sp = Tsp.basic.sp_ones_like(x_coord)
  x0 = rho_i*T.cos(theta_i)*one_sp
  y0 = rho_i*T.sin(theta_i)*one_sp
  patch_i = Tsp.sqr(x_coord-x0)+Tsp.sqr(y_coord-y0)
  offset_patch_i = Tsp.structured_add(-1.0*patch_i,offset)
  dense_patch_i = patch_i.toarray()
  offset_dense_patch_i = offset_patch_i.toarray()
  pmax = T.max(dense_patch_i,axis=1)
  pmin = offset - T.max(offset_dense_patch_i,axis=1)
  patch_i = Tsp.basic.row_scale(patch_i,-100.0/(pmin+0.05*(pmax-pmin)))
  ######patch_i = Tsp.structured_exp((-1.0/sigma)*(Tsp.sqr(x_coord-x0)+Tsp.sqr(y_coord-y0)))
  patch_i = Tsp.structured_exp(patch_i)
  patch_i = Tsp.basic.clean(patch_i)
  patch_i = Tsp.basic.row_scale(patch_i,1.0/(1e-38+Tsp.basic.sp_sum(patch_i,axis=1)))
  patch_i = Tsp.basic.row_scale(patch_i,1.0/(1e-38+Tsp.basic.sp_sum(patch_i,axis=1))) # 2 times for small number like 1e-40
  return patch_i
disk = []
for i in xrange(ntheta * nrho):
  disk.append(cal_patch(theta[i], rho[i], x_local, y_local))
disk = Tsp.basic.vstack(disk,format='csc')

cal_disk = theano.function(inputs=[x_local,y_local], outputs=[disk], on_unused_input='ignore')




import os
import sys

bg = int(sys.argv[1])
ed = int(sys.argv[2])

src_path = '/home/chenzhixing/data/BU3D/submesh/coords'
dst_path = '/home/chenzhixing/data/BU3D/submesh/disks/0_patch_8x2_50'

fnames = [fn for fn in os.listdir(src_path) if fn.endswith('.npz')]

for i in xrange(bg,ed):
  fn = fnames[i]
  t = np.load(os.path.join(src_path,fn))
  col = t['col'].astype(np.int32)
  ptr = t['ptr'].astype(np.int32)
  x = t['x_local'].astype(np.float32)
  y = t['y_local'].astype(np.float32)
  dim = ptr.shape[0]-1
  
  x_mat = csr_matrix((x, col, ptr),shape=(dim,dim))
  y_mat = csr_matrix((y, col, ptr),shape=(dim,dim))
  
  out_disk = cal_disk(x_mat,y_mat)[0]
  np.savez(os.path.join(dst_path,fn), indices=out_disk.indices, indptr=out_disk.indptr, data=out_disk.data, shape=out_disk.shape)
  print('%d done.(%d %d)' % (i,bg,ed))

#pp = np.load('tmp.npz')
#indices = pp['indices'].astype(np.int32)
#indptr = pp['indptr'].astype(np.int32)
#data = pp['data'].astype(np.float32)
#shape = pp['shape'].astype(np.int32)
#
#pp_mat = csc_matrix((data, indices, indptr),shape=shape)
#print pp_mat
