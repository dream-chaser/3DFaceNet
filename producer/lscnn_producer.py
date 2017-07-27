#
# This code is part of DeepShape <...>
#
# Copyright 2015
# Jonathan Masci
# <jonathan.masci@gmail.com>

import os
import sys
sys.path.append("..")
import zmq
from proto import training_sample_lscnn_pb2
import numpy as np
import time
import argparse
import scipy.sparse as sp
#from multiprocessing import Pool
from multiprocessing.dummy import Pool
from layers.utils import filtered_list_dir
import h5py

label_name_begin = -34
label_name_end   = -32

def load_mat_field_as_int(path, field):
  """
  path : string : absolute path to the mat file to load
  field : string : field to return from the dictionary
  return : np.array (float32)
  """
  data = h5py.File(path,'r')
  return np.asarray(data[field]).astype(np.int32).T

def load_mat_field(path, field):
  """
  path : string : absolute path to the mat file to load
  field : string : field to return from the dictionary
  return : np.array (float32)
  """
  if field == 'label':
    class_name = path[label_name_begin:label_name_end]
    if class_name == 'AN':
      label = [1,0,0,0,0,0,0]
    elif class_name == 'DI':
      label = [0,1,0,0,0,0,0]
    elif class_name == 'FE':
      label = [0,0,1,0,0,0,0]
    elif class_name == 'HA':
      label = [0,0,0,1,0,0,0]
    elif class_name == 'NE':
      label = [0,0,0,0,1,0,0]
    elif class_name == 'SA':
      label = [0,0,0,0,0,1,0]
    elif class_name == 'SU':
      label = [0,0,0,0,0,0,1]
    return np.asarray(label).astype(np.float32).T
  else:
    data = h5py.File(path,'r')
    return np.asarray(data[field]).astype(np.float32).T

def load_coord_mat_field(path):
  """
  path : string : absolute path to the mat file to load
  field : string : field to return from the dictionary
  return : 
  """
  data = np.load(path)
  col = data['col'].astype(np.int32)
  ptr = data['ptr'].astype(np.int32)
  x_value = data['x_local'].astype(np.float32)
  y_value = data['y_local'].astype(np.float32)
  return [x_value, y_value, col, ptr]

def load_disk_field(path, fn):
  """
  path : string : absolute path to the mat file to load
  field : string : field to return from the dictionary
  return : 
  """
  dir_list = os.listdir(path)
  disks = []
  cur_ind = 0
  disk_begin = []
  disk_end = []
  cur_i = 0
  dic = {}
  for d in dir_list:
    dir_info = d.split('_')
    if eval(dir_info[0]) == 0:
      continue
    for i in range(eval(dir_info[0])):
      dic[eval(dir_info[i+1])] = cur_i
    cur_i += 1

    disk = np.load(os.path.join(path,d,fn))
    indices = disk['indices'].astype(np.int32)
    indptr = disk['indptr'].astype(np.int32)
    data = disk['data'].astype(np.float32)
    shape = disk['shape'].astype(np.int32)
    disks.append(sp.csc_matrix((data,indices,indptr),shape=shape))
    disk_begin.append(cur_ind)
    cur_ind += shape[0]
    disk_end.append(cur_ind)
  res = sp.vstack(disks,format='csr')
  layer_begin = range(len(dic))
  layer_end = range(len(dic))
  for key in dic:
    layer_begin[key] = disk_begin[dic[key]]
    layer_end[key] = disk_end[dic[key]]
  return [res.data, res.indices, res.indptr, np.array(layer_begin).astype(np.int32), np.array(layer_end).astype(np.int32)]

def load_shape_field(path, field):
  """
  By design shapes are contained in a struct (from Matlab)
  called shape.
  This function loads the shape .mat file and returns the
  requested field.

  path : string : absolute path to the mat file for the shape
  field : string : field in the dict shape to return
  return : np.array (float32)
  """
  data = h5py.File(path, 'r')
  return np.asarray(data['shape'][field]).astype(np.float32).T

def producer(args):
  """
  This function takes care of data loading, sampling of the pairs (positive
  and negatives) and to put data, serialized with protobuf in the zmq queue.

  args.queue_size : int : number of data samples to store in the queue.
      Remember to set this value accordingly in the streamer as well
  args.outport1 : string : port number where to find the train streamer queue
  args.outport2 : string : port number where to find the validation streamer queue
  args.desc_dir : string : absolute path to the input descriptors (e.g.
      GEOVEC, WKS, etc). A matrix of size N x K is needed where N is the
      number of vertices in the shape and K is the number of dimensions of the
      descriptor.
  args.shape_dir : string : absolute path to the input shapes. The structure
      needs to contain the following fields:
          - Phi : matrix of laplace-beltrami eigenfunctions, e.g. matrix of
            size N x D, where D is the number of eigenfunctions to use.
            100 is what we use in the paper.
          - A : area vector
          - Lambda : eigenvalues
  args.batch_size : int : how many vertices to take. -1 take the entire shape
      and is recommended. LSCNN needs all points to be computed correctly,
      subsampling works with stochastic optimization but beware it is an
      approximation of the original LSCNN net model.
  """
  context = zmq.Context()
  zmq_socket = context.socket(zmq.PUSH)
  zmq_socket.sndhwm = args.queue_size
  zmq_socket.rcvhwm = args.queue_size
  if args.mode == 'train':
    zmq_socket.connect("tcp://127.0.0.1:" + args.outport1)
  elif args.mode == 'valid':
    zmq_socket.connect("tcp://127.0.0.1:" + args.outport2)

  #fnames = filtered_list_dir(args.shape_dir, ['mat'])

  ldm_num = 68
  fnames = []
  dir_file = open('./720_dir.txt','r')
  for line in dir_file:
    line = line.strip()
    for i in xrange(ldm_num):
      fnames.append(line[:-4]+('-%02d.mat' % i))

  fnames.sort()
  prm = np.fromfile("prm.bin", dtype = np.int32)
  ndata = len(prm)
  prm_i = args.prm_i
  cross_x = args.cross_x
  ndata_x = ndata/cross_x 
  prm_valid = prm[prm_i*ndata_x:(prm_i+1)*ndata_x]
  prm_train = np.append(prm[0:prm_i*ndata_x], prm[(prm_i+1)*ndata_x:])
  
  if args.mode == 'train':
    prm = prm_train
    print "Starting train sampling"
  elif args.mode == 'valid':
    prm = prm_valid
    print "Starting valid sampling"
  prm_len = len(prm)

#  expressions = ['AN', 'DI', 'FE', 'HA', 'SA', 'SU']
#  fnames = []
#  file_list = os.listdir(args.desc_dir)
#  for i in xrange(prm_len):
#    if prm[i] > 56:
#      prm[i] = prm[i] - 56
#      prefix = 'M' + ('%04d' % prm[i])
#    else:
#      prefix = 'F' + ('%04d' % prm[i])
#    for exp in expressions:
#      prefix_3 = prefix + '_' + exp + '03'
#      prefix_4 = prefix + '_' + exp + '04'
#      fnames.extend([fn for fn in file_list if fn.startswith(prefix_3)])
#      fnames.extend([fn for fn in file_list if fn.startswith(prefix_4)])
#  prm_len = len(fnames)
#  prm = np.random.permutation(prm_len)

  if args.alltomem == True:
    print "Loading all data into memory, hope it will fit!"
    descs = []
    labels = []
    #ldm_indices = []
    disks = []
    for i in xrange(prm_len):
      if i%20 == 0:
        print "%s: %d/%d" % (args.mode, i, prm_len)
      f = fnames[prm[i]][:-4]
      labels.append(load_mat_field(os.path.join(args.shape_dir, f+'.mat'), 'label'))
      #ldm_indices.append(load_mat_field_as_int(os.path.join(args.shape_dir, f), 'ldm_index').flatten())
      disks.append(load_disk_field(args.disk_dir, f+'.npz'))
      if args.const_fun:
        descs.append(np.ones((As[0].shape[0], 1)).astype(np.float32))
      else:
        descs.append(np.hstack([load_shape_field(os.path.join(args.shape_dir, f+'.mat'), 'X'),
                                load_shape_field(os.path.join(args.shape_dir, f+'.mat'), 'Y'),
                                load_shape_field(os.path.join(args.shape_dir, f+'.mat'), 'Z')]))
        #descs.append(load_mat_field(os.path.join(args.desc_dir, f+'.mat'), 'desc'))
    get_data = lambda x : [descs[x], labels[x], disks[x]]
    print "done"
  else:
    assert(args.const_fun == False) # TODO implement this case
    get_data = lambda x : [
            np.hstack([load_shape_field(os.path.join(args.shape_dir, fnames[x][:-4]+'.mat'), 'X'),
                       load_shape_field(os.path.join(args.shape_dir, fnames[x][:-4]+'.mat'), 'Y'),
                       load_shape_field(os.path.join(args.shape_dir, fnames[x][:-4]+'.mat'), 'Z')]),
            #load_mat_field(os.path.join(args.desc_dir, fnames[x][:-4]+'.mat'), 'desc'),
            load_mat_field(os.path.join(args.shape_dir, fnames[x][:-4]+'.mat'), 'label'),
            #load_mat_field_as_int(os.path.join(args.shape_dir, fnames[x]), 'ldm_index').flatten(),
            load_disk_field(args.disk_dir, fnames[x][:-4]+'.npz') ]
  
#  count = 0
#  ndots = 0
  data_index = 0
  while True:
    if args.alltomem == True:
      t_index = data_index
    else:
      t_index = prm[data_index]
    # TODO(czx): batch ?
    f, label, disk = get_data(t_index)
    print(f,f.shape)
    #print disk[1],disk[2],disk[3],disk[4]
    #print disk[1].shape,disk[2].shape,disk[3].shape,disk[4].shape
    #print type(disk[1][0]),type(disk[2][0]),type(disk[3][0]),type(disk[4][0])

    tr = training_sample_lscnn_pb2.training_sample_lscnn()

    tr.x           = f.tostring()
    tr.x_shape.extend(f.shape)
    tr.label       = label.tostring()
    tr.label_shape.extend(label.shape)
    #tr.ldm_index   = ldm_index.tostring()
    #tr.ldm_index_shape.extend(ldm_index.shape)

    tr.disk_data     = disk[0].tostring()
    tr.disk_data_shape.extend(disk[0].shape)
    tr.disk_indices  = disk[1].tostring()
    tr.disk_indices_shape.extend(disk[1].shape)
    tr.disk_indptr   = disk[2].tostring()
    tr.disk_indptr_shape.extend(disk[2].shape)
    tr.layer_begin   = disk[3].tostring()
    tr.layer_begin_shape.extend(disk[3].shape)
    tr.layer_end     = disk[4].tostring()
    tr.layer_end_shape.extend(disk[4].shape)
    tr.filename      = fnames[t_index][:-4]

    zmq_socket.send(tr.SerializeToString())
    
    data_index += 1
    if data_index == prm_len:
      data_index = 0

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generate data for training \
          LSCNN networks.')
  parser.add_argument('--desc_dir', metavar='desc_dir', type=str,
          dest='desc_dir', required=True,
          help='Where to find the descriptors')
  parser.add_argument('--shape_dir', metavar='shape_dir', type=str,
          dest='shape_dir', required=True,
          help='Where to find the shapes with Phi, A and Lambda fields')
  parser.add_argument('--disk_dir', metavar='disk_dir', type=str,
          dest='disk_dir', required=True,
          help='Where to find the calculated disks')
  parser.add_argument('--const_fun', metavar='const_fun', type=int,
          dest='const_fun',
          required=True)
  parser.add_argument('--nthreads', metavar='nthreads', type=int,
          dest='nthreads',
          required=True)
  parser.add_argument('--alltomem', metavar='alltomem', type=int,
          dest='alltomem',
          required=True)
  parser.add_argument('--batch_size', metavar='batch_size', type=int,
          dest='batch_size',
          help='Batch size',
          required=True)
  parser.add_argument('--queue_size', metavar='queue_size', type=int,
          dest='queue_size',
          help='Maximum size of the queue',
          required=True)
  parser.add_argument('--outport1', metavar='outport1', type=str,
          dest='outport1',
          help='Port to send train data',
          required=False, default='5579')
  parser.add_argument('--outport2', metavar='outport2', type=str,
          dest='outport2',
          help='Port to send validation data',
          required=False, default='5581')
  parser.add_argument('--mode', metavar='mode', type=str,
          dest='mode',
          help='produce mode',
          required=False, default='train')
  parser.add_argument('--prm_i', metavar='prm_i', type=int,
          dest='prm_i',
          required=True)
  parser.add_argument('--cross_x', metavar='cross_x', type=int,
          dest='cross_x',
          required=True)

  args = parser.parse_args()
  print "Generating data with %i threads" % args.nthreads
  producer(args)
  pool = Pool(args.nthreads)
  args = [args] * args.nthreads
  pool.map(producer, args)
  pool.join()
  pool.close()

