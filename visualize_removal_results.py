# --------------------------------------------------------
# Pytorch Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Rakshith Shetty, based on code by Jiasen Lu, Jianwei Yang and Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from tqdm import tqdm as tqdm
from removalcode.object_remover import ObjectRemover
from torchvision.transforms.functional import resize
from collections import defaultdict
import matplotlib.pyplot as plt
from os.path import basename, exists, join, splitext
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def saveIndividImages(image_list, nameList, sample_dir, fp, cls):
    #sample_dir = join(params['sample_dump_dir'], basename(params['model'][0]).split('.')[0])
    fdir = join(sample_dir, splitext(basename(fp))[0]+'_'+cls)
    if not exists(fdir):
        os.makedirs(fdir)

    for i, img in enumerate(image_list):
        fname = join(fdir, nameList[i]+'.png')
        cv2.imwrite(fname, img)
        print('Saving into file: ' + fname)



def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--use_augmented', dest='use_augmented',
                      help='Should we test on removal augmented data',
                      default=0, type=int)
  parser.add_argument('--removal_dilation', dest='removal_dilation',
                      help='Should we test on removal augmented data',
                      default=11, type=int)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--save_append', dest='save_append',
                      help='directory to load models', default="",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      default = True,
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checkpoint_file', dest='checkpoint_file',
                      help='override all the directory params',
                      default=None, type=str)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--savesepimgs', dest='savesepimgs',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--result_file', dest='result_file',
                      help='override all the directory params',
                      default=None, type=str)
  parser.add_argument('--eval_file', dest='eval_file',
                      help='override all the directory params',
                      default=None, type=str)
  parser.add_argument('--plot_type', dest='plot_type',
                      help='override all the directory params',
                      default='rem', type=str)

  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def loadAndPrepImg(data, box=[], catName = ''):
  im_data.data.resize_(data[0].size()).copy_(data[0])
  im_info.data.resize_(data[1].size()).copy_(data[1])
  if cfg.REMOVE_AUGMENT:
      rem_mask.data.resize_(data[4].size()).copy_(data[4])
      new_img = remover(im_data[None,::], rem_mask[None,::])
  else:
      new_img = im_data

  im = cv2.resize(new_img[0].data.cpu().numpy().transpose(1,2,0)+cfg.PIXEL_MEANS,(int(im_info[1]/im_info[2]),int(im_info[0]/im_info[2])))#cv2.imread(imdb.image_path_at(i))
  #im2 = cv2.imread(imdb.image_path_at(i))
  im2show = np.copy(im)
  if len(box):
    im2show = vis_detections(im2show, catName, box, 0.3)

  return im2show


if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_12":
      args.imdb_name = "voc_2012_trainval"
      args.imdbval_name = "voc_2012_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival" if not args.use_augmented else "coco_2014_augmentedFilteredminival"
      #args.imdbval_name = "coco_2014_minival" if not args.use_augmented else "coco_2014_augmentedminival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  if args.use_augmented:
      cfg.REMOVE_AUGMENT = True
      remover = ObjectRemover(dilateMask=args.removal_dilation, pixel_mean=cfg.PIXEL_MEANS)
      remover.cuda()
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)
  rem_mask= torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()
    rem_mask = rem_mask.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  #if args.cuda:
  #  fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  num_images = len(imdb.image_index)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  #fasterRCNN.eval()

  #all_boxes = pickle.load(open(args.result_file, 'rb'))
  d_imgId_toIndex = {rdb['img_id']:i for i,rdb in enumerate(dataset._roidb)}

  resEv = pickle.load(open(args.eval_file, 'rb'))
  print('Loaded evaluation results')
  img_save_dir = os.path.join('removal_results', args.eval_file.split('/')[-2], args.plot_type)
  if not os.path.isdir(img_save_dir):
    os.makedirs(img_save_dir)

  catToIndex = {cid:i for i, cid in enumerate(resEv.params.catIds)}
  imidToIndex = {imid:i for i, imid in enumerate(resEv.params.imgIds)}
  bimidToMaxDetectScores = defaultdict(float)
  bimidToMaxDetectBox = defaultdict(list)
  beforeAfters = []
  pltIds = []
  pltCats = []
  bad_categ = set(resEv.cocoGt.getCatIds(['dining table', 'bed']))
  for i,cid in enumerate(resEv.params.catIds):
    for j,imid in enumerate(resEv.params.imgIds):
      idx = i*len(resEv.params.imgIds) * 4 + j
      if (resEv.evalImgs[idx]) is not None:
        assert(resEv.evalImgs[idx]['category_id'] == cid)
        bimidToMaxDetectScores[imid,cid] = resEv.evalImgs[idx]['dtScores'][0] if len(resEv.evalImgs[idx]['dtScores']) else 0.
        bimidToMaxDetectBox[imid,cid] = resEv.evalImgs[idx]['dtIds'] if len(resEv.evalImgs[idx]['dtScores']) else []

  if args.plot_type == 'rem':
    for imid in resEv.params.imgIds:
        if '_' in str(imid):
            base_id = imid.split('_')[0]
            cat_id  = int(imid.split('_')[1])
            if cat_id not in bad_categ:
                beforeAfters.append([bimidToMaxDetectScores[base_id, cat_id], bimidToMaxDetectScores[imid,cat_id]])
                pltIds.append(imid)
                pltCats.append(cat_id)

  elif args.plot_type == 'rem_oth':
    imidToGtClass = {rdb['img_id']: list(set(rdb['gt_classes'])) for rdb in dataset._roidb}
    imidToGtClass = {imid: resEv.cocoGt.getCatIds([imdb._classes[cind] for cind in imidToGtClass[imid]]) if len(imidToGtClass[imid]) else [] for imid in imidToGtClass}
    for imid in resEv.params.imgIds:
        if '_' in str(imid):
            rem_cat = int(imid.split('_')[1])
            base_id = imid.split('_')[0]
            if rem_cat not in bad_categ:
              for cat_id in imidToGtClass[imid]:
                  if cat_id !=rem_cat:
                      #if (bimidToMaxDetectScores[imid,cat_id] >0.3) or (bimidToMaxDetectScores[base_id,cat_id] > 0.3):
                      beforeAfters.append([bimidToMaxDetectScores[base_id, cat_id], bimidToMaxDetectScores[imid,cat_id]])
                      pltIds.append(imid)
                      pltCats.append(cat_id)
  else:
      print('Unknown plot type')
      assert(0)


  def onpick(event):
      ind = event.ind
      imgId = pltIds[ind[0]]
      oImgId = imgId.split('_')[0]
      catId = pltCats[ind[0]]
      catname = resEv.cocoGt.cats[catId]['name']
      removedcat = resEv.cocoGt.cats[int(imgId.split('_')[1])]['name']
      print('Image id %s, cat: %s, removed cat: %s'%(imgId, catname, removedcat))

      # Load and prep the edited image
      dImgInd = d_imgId_toIndex[imgId]
      if len(bimidToMaxDetectBox[imgId,catId]):
        remBoxData = resEv.cocoDt.loadAnns(bimidToMaxDetectBox[imgId,catId])
        remBox = np.concatenate([np.array(ann['bbox'] + [ann['score']])[None,:] for ann in remBoxData],axis=0)
        remBox[:,2] = remBox[:,0] + remBox[:,2]
        remBox[:,3] = remBox[:,1] + remBox[:,3]
      else:
        remBox = []

      remImg = loadAndPrepImg(dataset[dImgInd], remBox, catname)

      # Load and prep the edited image
      oImgInd = d_imgId_toIndex[oImgId]
      if len(bimidToMaxDetectBox[oImgId,catId]):
        oBoxData = resEv.cocoDt.loadAnns(bimidToMaxDetectBox[oImgId,catId])
        if len(oBoxData):
            oBox = np.concatenate([np.array(ann['bbox'] + [ann['score']])[None,:] for ann in oBoxData],axis=0)
            oBox[:,2] = oBox[:,0] + oBox[:,2]
            oBox[:,3] = oBox[:,1] + oBox[:,3]
        else:
            oBox = []
      else:
        oBox = []

      oImg = loadAndPrepImg(dataset[oImgInd], oBox, catname)

      # Load and prep the edited image

      #cv2.imwrite('result.png', im2show)
      #pdb.set_trace()
      concatImg = np.concatenate([oImg,remImg],axis=1).astype(np.uint8)
      cv2.imshow('concat',  concatImg)
      #cv2.imshow('original', oImg.astype(np.uint8))
      keyInp = cv2.waitKey(0)
      cv2.destroyWindow('concat')
      if (keyInp & 0xFF == ord('s')):
          if args.savesepimgs:
            saveIndividImages([oImg,remImg],['orignal','edited'], img_save_dir, oImgId, removedcat+'_'+catname)
          else:
            cv2.imwrite(os.path.join(img_save_dir,oImgId+'_'+catname+'.png'), concatImg)
      return


  fig = plt.figure();ax = plt.subplot();
  fig.canvas.mpl_connect('pick_event', onpick)
  beforeAfters = np.array(beforeAfters)
  cax = ax.scatter(np.log(beforeAfters[:,0]+1e-3), np.log(beforeAfters[:,1]+1e-3), alpha=0.5,c=pltCats,cmap=plt.cm.plasma,s=14, picker=True)
  ax.plot(np.arange(np.log(beforeAfters[:,0]+1e-3).min(), np.log(beforeAfters[:,0]+1e-3).max()),np.arange(np.log(beforeAfters[:,0]+1e-3).min(), np.log(beforeAfters[:,0]+1e-3).max()), 'k-',linewidth=2)
  fig.colorbar(cax);

  ax.set_xlabel('Classifier score original image', fontsize=14);
  ax.set_ylabel('Classifier score modified image',fontsize=14);
  plt.show()



  end = time.time()
  print("test time: %0.4fs" % (end - start))


