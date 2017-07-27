import theano
import theano.tensor as T
import theano.sparse as Tsp
import numpy as np
import scipy.sparse as sp

a = T.fmatrix('a')
b = Tsp.csc_matrix('b')
c = Tsp.basic.dot(a,b)
test = theano.function(inputs=[a,b], outputs=[c], on_unused_input='warn')
aa = [[-1,1,1]]
aa = np.array(aa).astype(np.float32)
row = [0,1,2]
col = [0,1,2]
data = [1,-1,-1]
bb = sp.csc_matrix((data, (row,col))).astype(np.float32)
x = test(aa,bb)
print x
