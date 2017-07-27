"""
Script to perform training and saving of shape features with 
a localized spectral convolutional network (LSCNN).
"""

import os
import sys
import time
import zmq
import random
from proto import training_sample_lscnn_pb2
import numpy as np
import scipy.misc
import argparse
from sklearn.metrics import roc_auc_score, precision_recall_curve
import logging
import yaml
import theano
import theano.tensor as T
import theano.sparse as Tsp
from collections import OrderedDict
from itertools import izip
import scipy.sparse as sp
import scipy.io
import h5py

from layers.layers_lscnn import *
from layers.losses import softmax_loss
from model.lscnn import LSCNN
from optim.optim import adadelta_updates, train
from perfeval.AUC_monitor import PerformanceMonitor
from scipy.sparse import csr_matrix

ldm_num = 68

def parse_input_stack(x, xs):
  x = np.fromstring(x, dtype=np.float32).reshape(xs)
  return x

def parse_input_stack_as_int(x, xs):
  x = np.fromstring(x, dtype=np.int32).reshape(xs)
  return x

def fetch_data(queue_size, inport):
  context = zmq.Context()
  # recieve work
  consumer_receiver = context.socket(zmq.PULL)
  consumer_receiver.sndhwm = queue_size
  consumer_receiver.rcvhwm = queue_size
  consumer_receiver.connect("tcp://127.0.0.1:" + inport)

  tr = training_sample_lscnn_pb2.training_sample_lscnn()
  while True:
    x = []
    disk = []
    layer_begin = []
    layer_end = []
    fname = ''
    check = [-1 for ck in range(ldm_num)]
    i = 0
    while True:
      if i == ldm_num:
        break
      data = consumer_receiver.recv()
      tr.ParseFromString(data)

      if i == 0: 
        fname = tr.filename[:-2]
        label = parse_input_stack(tr.label, tr.label_shape).flatten()
      elif tr.filename[:-2] != fname:
        print('fetch_data warning: %s %s re-fetch...' % (fname, tr.filename[:-2]))
        i = 0
        fname = tr.filename[:-2]
        label = parse_input_stack(tr.label, tr.label_shape).flatten()
        x = []
        disk = []
        layer_begin = []
        layer_end = []
        check = [-1 for ck in range(ldm_num)]
      
      xi = parse_input_stack(tr.x, tr.x_shape)
      x.append(xi)
      #ldm_index = parse_input_stack_as_int(tr.ldm_index, tr.ldm_index_shape).flatten()
  
      disk_data = parse_input_stack(tr.disk_data, tr.disk_data_shape)
      disk_indices = parse_input_stack_as_int(tr.disk_indices, tr.disk_indices_shape)
      disk_indptr = parse_input_stack_as_int(tr.disk_indptr, tr.disk_indptr_shape)
      disk.append(csr_matrix((disk_data, disk_indices, disk_indptr),shape=(disk_indptr.shape[0]-1,xi.shape[0])))
  
      layer_begin.append(parse_input_stack_as_int(tr.layer_begin, tr.layer_begin_shape))
      layer_end.append(parse_input_stack_as_int(tr.layer_end, tr.layer_end_shape))
      
      if tr.filename[-2] == '0':
        part_i = eval(tr.filename[-1:])
      else:
        part_i = eval(tr.filename[-2:])
      check[i] = part_i
      i+=1

    for ck in range(ldm_num):
      if check[ck] != ck:
        print('fetch_data error: %s' % fname)
        exit(-2)

    yield ([np.array(x).astype(np.float32), label, sp.vstack(disk), np.array(layer_begin).astype(np.int32), np.array(layer_end).astype(np.int32)])


if __name__ == "__main__":
  sys.setrecursionlimit(10**6)

  parser = argparse.ArgumentParser(description='Trains a windowed spectral network.')
  parser.add_argument('--config_file', metavar='config_file', type=str,
          dest='config_file', required=True,
          help='Experiment configuration file, see test_yaml.yaml for reference.')
  parser.add_argument('--mode', metavar='mode', type=str, dest='mode',
          help='train and dump modes available',
          required=True)
  parser.add_argument('--model_to_load', metavar='model_to_load', type=str,
          dest='model_to_load',
          required=False, default=None)
  parser.add_argument('--output_dump', metavar='output_dump', type=str,
          dest='output_dump',
          required=False, default=None)
  parser.add_argument('--queue_size', metavar='queue_size', type=int,
          dest='queue_size',
          help='Maximum size of the queue',
          required=True)
  parser.add_argument('--l2reg', metavar='l2reg', type=float,
          dest='l2reg',
          required=True)
  parser.add_argument('--inport', metavar='inport', type=str,
          dest='inport',
          help='Port from which to collect data)',
          required=False, default='5580')
  parser.add_argument('--desc_dir', metavar='desc_dir', type=str,
          dest='desc_dir',
          required=False, default=None)
  parser.add_argument('--shape_dir', metavar='shape_dir', type=str,
          dest='shape_dir',
          required=False, default=None)
  parser.add_argument('--valid', metavar='valid', type=int, dest='valid',
          help='is validation or not',
          required=False, default=0)
  
  args = parser.parse_args()

  #iterator_valid = fetch_data(args.queue_size, "5582")
  #tmp = iterator_valid.next()
  #print tmp[0].shape
  #print tmp[1].shape
  #print tmp[2].shape
  #print tmp[3].shape
  #print tmp[4].shape
  #exit()

  f = file(args.config_file, 'r')
  conf = yaml.load(f)
  f.close()
  
  exp_name = "%s_l2reg_%f" % (os.path.split(args.config_file)[1].split('.')[0], + args.l2reg)
  print("Experiment name: %s" % exp_name)
  conf['out_path'] = os.path.join(conf['out_path'], exp_name)
  if not os.path.isdir(conf['out_path']):
    os.makedirs(conf['out_path'])

  logging.basicConfig(filename=os.path.join(conf['out_path'], '%s-log.txt' % exp_name), 
      level=logging.DEBUG,
      format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%Y/%m/%d %I:%M:%S %p')
  logging.getLogger().addHandler(logging.StreamHandler())
  logging.info('Experiment starts')
  logging.info('Saving experiment data to %s', conf['out_path'])

  logging.info("Setting random state...")
  start_time = time.clock()
  rng = conf['rng'] 
  srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
  np.random.seed(conf['seed'])
  logging.info("...done %f" % (time.clock() - start_time))

  logging.info("Creating the model...")
  start_time = time.clock()

  x  = T.ftensor3('x')
  label = T.fvector('label')
  #ldm_index = T.ivector('ldm_index')
  disk = Tsp.csr_matrix('disk')
  layer_begin = T.imatrix('layer_begin')
  layer_end = T.imatrix('layer_end')

  model = LSCNN(rng,
          conf['layers'],
          conf['drop'])

  model.inputs = [x, label, disk, layer_begin, layer_end]
  model.fwd_inputs = [x, label, disk, layer_begin, layer_end]
  model.w_constraints = eval(conf['w_constraints']) 
  logging.info("...done %f" % (time.clock() - start_time))

  logging.info("Checking if there is already a best model to load...")
  start_time = time.clock()
  if args.model_to_load is not None:
    logging.info("...loading model %s" % args.model_to_load)
    model.load(args.model_to_load)
  else:
    if os.path.isfile(os.path.join(conf['out_path'], "%s-best-model.pkl" % exp_name)):
      logging.info("...loading best model")
      model.load(os.path.join(conf['out_path'], "%s-best-model.pkl" % exp_name))
    else:
      logging.info("...saving initial model")
      model.save(os.path.join(conf['out_path'], "%s-init-model" % exp_name))
      model.save(os.path.join(conf['out_path'], "%s-last-model" % exp_name))
  logging.info("...done %f" % (time.clock() - start_time))

  logging.info("Cost definition...")
  start_time = time.clock() 
  out = model.fwd(x, label, disk, layer_begin, layer_end) # TODO(czx): what is the usage of masks, cut the dropout!!!
  cost = softmax_loss(out, label)

  for p in model.get_params():
    cost += (args.l2reg * T.sqr(p)).mean()

  model.cost = cost   
  logging.info("...done %f" % (time.clock() - start_time)) 

  if args.mode == "train":
    logging.info("Starting training...")
    start_time = time.clock()
    train(model, cost, PerformanceMonitor(), 
          conf,
          conf['out_path'], exp_name, 
          conf['iter_eval'], logging, args.queue_size, args.inport, 
          fetch_data=fetch_data, n_epochs=conf['n_epochs'], is_validation=args.valid,
          n_batches_valid=conf['n_batches_valid'], n_batches_train=conf['n_batches_train'])
    logging.info("...done training %f" % (time.clock() - start_time))
#  elif args.mode == "dump": # TODO(czx): not yet modified
#    out_dir = os.path.join(args.output_dump, exp_name)
#    if not os.path.isdir(out_dir):
#      print "Creating %s" % out_dir
#      os.makedirs(out_dir)
#
#    model_out = model.fwd(*model.fwd_inputs)[0]
#    fwd_model = theano.function(inputs=model.fwd_inputs,
#            outputs=model_out,
#            on_unused_input='warn')
#
#    fnames = filtered_list_dir(args.shape_dir, ['mat'])
#    for fn in fnames:
#      shape = h5py.File(os.path.join(args.shape_dir, fn), 'r')
#      V = np.asarray(shape['shape']['Phi']).astype(np.float32).T
#      A = np.diag(np.asarray(shape['shape']['A']).astype(np.float32)).flatten()
#      L = np.asarray(shape['shape']['Lambda']).astype(np.float32).flatten()
#      if args.desc_dir is None:
#        print "Dumping CONST function"
#        data = np.ones((V.shape[0], 1)).astype(np.float32)
#      else:
#        data = h5py.File(os.path.join(args.desc_dir, fn), 'r')
#        data = np.asarray(data['desc']).astype(np.float32).T
#
#      #desc = []
#      #for i in xrange(0,data.shape[0],100):
#      #  to = min(data.shape[0],i+100)
#      #  desc.append(fwd_model(data[i:to], V[i:to], A[i:to], L))
#      #desc = np.concatenate(desc, axis=0)
#
#      out = dict()
#      #out['desc'] = desc #fwd_model(data, V, A, L)
#      start_time = time.time()
#      out['desc'] = fwd_model(data, V, A, L)
#      print time.time() - start_time
#      fout = os.path.join(out_dir, '%s' % fn)
#      print "Saving desc in %s" % fout
#      scipy.io.savemat(fout, out)
#
