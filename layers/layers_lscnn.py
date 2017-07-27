import sys
import math
import theano.tensor
import theano.tensor as T
import numpy as np
import theano
import theano.sparse as Tsp
from utils import *

import theano.tensor.nnet.conv
from theano.sandbox.cuda.basic_ops import gpu_contiguous
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.cuda.basic_ops import gpu_from_host

class ILSCNNLayer:
  """ 
  Minimal interface for a LSCNN layer
  """
  
  def fwd(self, x, disk=None, layer_begin=None, layer_end=None):
    """
    x: input signal
    """
    raise NotImplementedError(str(type(self)) + " does not implement fwd.")

  def get_params(self):
    raise NotImplementedError(str(type(self)) + " does not implement get_params.")

  def set_params(self, w):
    raise NotImplementedError(str(type(self)) + " does not implement set_params.")   

class MaxLayer(ILSCNNLayer):
  def __init__(self, layer_name, activation=None):
    self.layer_name = layer_name
    self.activation = activation
    self.params = []

  def fwd(self, x, disk=None, layer_begin=None, layer_end=None):
    x = gpu_contiguous(x)
    if x.ndim == 4:
        x = x.flatten(2)
    lin_output = T.max(x,axis=0)

    return (lin_output if self.activation is None
                       else self.activation(lin_output))

  def get_params(self):
    return self.params

  def set_params(self, w):
    pass

class StatisticLayer(ILSCNNLayer):
  def __init__(self, rng, n_in, n_out, layer_name, activation=None):
    self.rng = rng
    self.n_in = n_in
    self.n_out = n_out
    self.layer_name = layer_name
    self.activation = activation

    W_values = np.asarray(rng.uniform(
               low=-np.sqrt(6. / (n_in + n_out)),
               high=np.sqrt(6. / (n_in + n_out)),
               size=(n_in, n_out)), dtype=theano.config.floatX)
    if activation == theano.tensor.nnet.sigmoid:
      W_values *= 4
    W = theano.shared(value=W_values, name=layer_name+'-W', borrow=True)

    b_values = np.zeros((n_out,), dtype=theano.config.floatX) + np.float32(0.5)
    b = theano.shared(value=b_values, name=layer_name+'-b', borrow=True)

    self.W = W
    self.b = b

    # parameters of the model
    self.params = [self.W, self.b]

  def fwd(self, x, disk=None, layer_begin=None, layer_end=None):
    x = gpu_contiguous(x)
    if x.ndim == 4:
        x = x.flatten(2)
    x_uniform = x - T.mean(x,axis=0)
    x_cov = T.dot(x_uniform.T,x_uniform).flatten(1)
    lin_output = T.dot(x_cov, self.W) + self.b

    return (lin_output if self.activation is None
                       else self.activation(lin_output))

  def get_params(self):
    return self.params

  def set_params(self, w):
    self.W.set_value(w[0])
    self.b.set_value(w[1])

class MLPLayer(ILSCNNLayer):
  def __init__(self, rng, n_in, n_out, layer_name, activation=None):
    self.rng = rng
    self.n_in = n_in
    self.n_out = n_out
    self.layer_name = layer_name
    self.activation = activation

    W_values = np.asarray(rng.uniform(
               low=-np.sqrt(6. / (n_in + n_out)),
               high=np.sqrt(6. / (n_in + n_out)),
               size=(n_in, n_out)), dtype=theano.config.floatX)
    if activation == theano.tensor.nnet.sigmoid:
      W_values *= 4
    W = theano.shared(value=W_values, name=layer_name+'-W', borrow=True)

    b_values = np.zeros((n_out,), dtype=theano.config.floatX) + np.float32(0.5)
    b = theano.shared(value=b_values, name=layer_name+'-b', borrow=True)

    self.W = W
    self.b = b

    # parameters of the model
    self.params = [self.W, self.b]

  def fwd(self, x, disk=None, layer_begin=None, layer_end=None):
    x = gpu_contiguous(x)
    if x.ndim == 4:
        x = x.flatten(2)
    lin_output = T.dot(x, self.W) + self.b

    return (lin_output if self.activation is None
                       else self.activation(lin_output))

  def get_params(self):
    return self.params

  def set_params(self, w):
    self.W.set_value(w[0])
    self.b.set_value(w[1])

#TODO: Var
class MLPLayer_FC(ILSCNNLayer):
  def __init__(self, rng, n_in, n_out, layer_name, activation=None):
    self.rng = rng
    self.n_in = n_in
    self.n_out = n_out
    self.layer_name = layer_name
    self.activation = activation

    W_values = np.asarray(rng.uniform(
               low=-np.sqrt(6. / (n_in + n_out)),
               high=np.sqrt(6. / (n_in + n_out)),
               size=(n_in, n_out)), dtype=theano.config.floatX)
    if activation == theano.tensor.nnet.sigmoid:
      W_values *= 4
    W = theano.shared(value=W_values, name=layer_name+'-W', borrow=True)

    b_values = np.zeros((n_out,), dtype=theano.config.floatX) + np.float32(0.5)
    b = theano.shared(value=b_values, name=layer_name+'-b', borrow=True)

    self.W = W
    self.b = b

    # parameters of the model
    self.params = [self.W, self.b]

  def fwd(self, x, disk=None, layer_begin=None, layer_end=None):
    x = gpu_contiguous(x)
    x = x.flatten(1)
    #x = T.reshape(x, (1, self.n_in))
    lin_output = T.dot(x, self.W) + self.b

    return (lin_output if self.activation is None
                       else self.activation(lin_output))

  def get_params(self):
    return self.params

  def set_params(self, w):
    self.W.set_value(w[0])
    self.b.set_value(w[1])

#TODO: Var
# This is landmark layer
class MLPLayer_last(ILSCNNLayer):
  def __init__(self, rng, n_in, n_out, layer_name, activation=None):
    self.rng = rng
    self.n_in = n_in
    self.n_out = n_out
    self.layer_name = layer_name
    self.activation = activation

    W_values = np.asarray(rng.uniform(
               low=-np.sqrt(6. / (n_in + n_out)),
               high=np.sqrt(6. / (n_in + n_out)),
               size=(n_in, n_out)), dtype=theano.config.floatX)
    if activation == theano.tensor.nnet.sigmoid:
      W_values *= 4
    W = theano.shared(value=W_values, name=layer_name+'-W', borrow=True)

    b_values = np.zeros((n_out,), dtype=theano.config.floatX) + np.float32(0.5)
    b = theano.shared(value=b_values, name=layer_name+'-b', borrow=True)

    self.W = W
    self.b = b

    # parameters of the model
    self.params = [self.W, self.b]

  def fwd(self, x, ldm_index, disk=None, layer_begin=None, layer_end=None):
    x = gpu_contiguous(x)
    x = x[ldm_index]
    x = x.flatten(1)
    #x = T.reshape(x, (1, self.n_in))
    lin_output = T.dot(x, self.W) + self.b

    return (lin_output if self.activation is None
                       else self.activation(lin_output))

  def get_params(self):
    return self.params

  def set_params(self, w):
    self.W.set_value(w[0])
    self.b.set_value(w[1])

class GCNNLayer(ILSCNNLayer):
  def __init__(self, rng, nin, nout, ntheta, nrho, layer_name, activation=relu):
    self.rng = rng
    self.nin = nin
    self.nout = nout
    self.ntheta = ntheta
    self.nrho = nrho
    self.layer_name = layer_name
    self.activation = activation
    self.layer_id = eval(layer_name[-1])

    #t0 = np.zeros(nrho,dtype=theano.config.floatX)
    #r0 = np.zeros(ntheta,dtype=theano.config.floatX)
    #theta_v = np.array([(tv*2+1)*math.pi/ntheta for tv in range(ntheta)],dtype=theano.config.floatX).reshape(ntheta,1) + t0
    #rho_v = np.array([(rv+0.5)*max_rho/nrho for rv in range(nrho)],dtype=theano.config.floatX).reshape(nrho,1) + r0
    
    #theta = theano.shared(value=theta_v.transpose().flatten(), name=layer_name+'-theta_op', borrow=True)
    #rho = theano.shared(value=rho_v.flatten(), name=layer_name+'-rho_op', borrow=True)
    #self.theta = theta
    #self.rho = rho

    a_values = np.asarray(rng.uniform(
               low=-np.sqrt(6. / (nin + nout)),
               high=np.sqrt(6. / (nin + nout)),
               size=(nout, nin, ntheta*nrho, 1)), dtype=theano.config.floatX)
    a = theano.shared(value=a_values, name=layer_name+'-a', borrow=True)
    self.a = a
      
    # parameters of the model
    self.params = [self.a]

  def fwd(self, x, disk=None, layer_begin=None, layer_end=None):
    """
    x : signal
    """
#    def cal_patch(theta_i, rho_i, x_dense, y_dense):
#      x_coord = Tsp.csr_from_dense(x_dense)
#      y_coord = Tsp.csr_from_dense(y_dense)
#      x0 = rho_i*T.cos(theta_i)*Tsp.basic.sp_ones_like(x_coord)
#      y0 = rho_i*T.sin(theta_i)*Tsp.basic.sp_ones_like(y_coord)
#      patch_i = Tsp.structured_exp((-1.0/self.sigma)*(Tsp.sqr(x_coord-x0)+Tsp.sqr(y_coord-y0)))
#      patch_i = Tsp.basic.row_scale(patch_i,1.0/Tsp.basic.sp_sum(patch_i,axis=1))
#      return patch_i.toarray()
#    
#    scan_results,scan_updates = theano.scan(fn=cal_patch, outputs_info=None,
#                                            sequences=[self.theta, self.rho],
#                                            non_sequences=[x_local.toarray(), y_local.toarray()])
#    disk = Tsp.csr_from_dense(T.swapaxes(scan_results,0,1).reshape([self.ntheta * self.nrho * x.shape[0], x.shape[0]]))
#    patch = Tsp.basic.structured_dot(disk,x)
#    patch = T.reshape(patch,(x.shape[0], self.ntheta*self.nrho, self.nin, 1))
#    patch = T.swapaxes(patch, 1, 2)


#    def cal_patch(theta_i, rho_i, x_coord, y_coord):
#      x0 = rho_i*T.cos(theta_i)*Tsp.basic.sp_ones_like(x_coord)
#      y0 = rho_i*T.sin(theta_i)*Tsp.basic.sp_ones_like(y_coord)
#      patch_i = Tsp.structured_exp((-1.0/self.sigma)*(Tsp.sqr(x_coord-x0)+Tsp.sqr(y_coord-y0)))
#      patch_i = Tsp.basic.row_scale(patch_i,1.0/(1e-30+Tsp.basic.sp_sum(patch_i,axis=1)))
#      return patch_i
#    disk = []
#    for i in xrange(self.ntheta * self.nrho):
#      disk.append(cal_patch(self.theta[i], self.rho[i], x_local, y_local))
#    disk = Tsp.basic.vstack(disk,format='csc')

    layer_disk = disk[layer_begin[self.layer_id]:layer_end[self.layer_id],:]
    patch = Tsp.basic.structured_dot(layer_disk,x)
    patch = T.reshape(patch,[self.ntheta*self.nrho, x.shape[0], x.shape[1]])
    patch = T.reshape(T.swapaxes(T.swapaxes(patch,0,1),1,2),[x.shape[0], self.nin, self.ntheta*self.nrho, 1])

    return self.activation(theano.tensor.nnet.conv.conv2d(patch, self.a).flatten(2))

  def get_params(self):
    return self.params

  def set_params(self, a):
    self.a.set_value(a[0])
