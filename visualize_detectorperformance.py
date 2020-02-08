# --------------------------------------------------------
# Pytorch Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Rakshith Shetty, based on code by Jiasen Lu, Jianwei Yang and Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import time
import cv2
import torch
from torch.autograd import Variable
import pickle
from tqdm import tqdm as tqdm
#from removalcode.object_remover import ObjectRemover
from torchvision.transforms.functional import resize
from collections import defaultdict
import matplotlib.pyplot as plt
from os.path import basename, exists, join, splitext
import pdb
from models import *
from utils.datasets import *
from utils.utils import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
  parser = argparse.ArgumentParser(prog='test.py')
  parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
  parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
  parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
  parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
  parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
  parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
  parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
  parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
  parser.add_argument('--img_size', type=int, default=None, help='size of each image dimension')

  parser.add_argument('--use_augmented', dest='use_augmented',
                      help='Should we test on removal augmented data',
                      default=0, type=int)
  parser.add_argument('--save_append', dest='save_append',
                      help='directory to load models', default="",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      default = True,
                      action='store_true')
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--savesepimgs', dest='savesepimgs',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--eval_file', dest='eval_file',
                      help='override all the directory params',
                      default=None, type=str)
  parser.add_argument('--res_file', dest='res_file',
                      help='override all the directory params',
                      default=None, type=str)
  parser.add_argument('--plot_type', dest='plot_type',
                      help='override all the directory params',
                      default='rem', type=str)
  parser.add_argument('--dtTh', dest='dtTh',
                      help='override all the directory params',
                      default=0.1, type=float)
  parser.add_argument('--showall', dest='showall',
                      help='override all the directory params',
                      default=0, type=int)
  parser.add_argument('--class', dest='class_name',
                      help='override all the directory params',
                      default='person', type=str)
  parser.add_argument('--area', dest='area',
                      help='override all the directory params',
                      default='all', type=str)

  args = parser.parse_args()
  print(args, end='\n\n')
  return args

colors = [[0,0,255]]+[[random.randint(0, 255) for _ in range(3)] for _ in range(90)]\


def loadAndPrepImg(data, box=[], catNames = '', th=0.5):
  im_data.data.resize_(data[0].size()).copy_(data[0])
  im_info = data[3]
  new_img = im_data

  #im = cv2.resize(new_img.data.cpu().numpy().transpose(1,2,0),(im_info[1], im_info[0]))[:,:,::-1]#cv2.imread(imdb.image_path_at(i))
  im = (new_img.data.cpu().numpy().transpose(1,2,0)[:,:,::-1]*255.).astype(np.uint8)#cv2.imread(imdb.image_path_at(i))
  #im2 = cv2.imread(imdb.image_path_at(i))
  im2show = im#np.copy(im)
  if len(box):
      for i in range(box.shape[0]):
        if box[i,4] > th or (i==0):
            im2show = plot_one_box(box[i,:4], im2show, label=catNames[i]+'  '+"{:.2f}".format(box[i,4]), color=colors[i%len(colors)], line_thickness=1 if i > 0 else 2)

  return im2show

def l2box(bb1, bb2, imsz ):
    bc1 = [(bb1[0] + bb1[2]//2)/imsz[0], (bb1[1] + bb1[3]//2)/imsz[1]]
    bc2 = [(bb2[0] + bb2[2]//2)/imsz[0], (bb2[1] + bb2[3]//2)/imsz[1]]
    dist = np.sqrt((bc1[0] - bc2[0])**2. + (bc1[1] - bc2[1])**2.)
    return dist

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


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

  # Configure run
  data_cfg = parse_data_cfg(args.data_cfg)
  test_path = data_cfg['valid']
  # if (os.sep + 'coco' + os.sep) in test_path:  # COCO dataset probable
  #     save_json = True  # use pycocotools

  # Dataloader
  dataset = LoadImagesAndLabels(test_path, img_size=args.img_size)

  seen = 0
  print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
  mP, mR, mAP, mAPj = 0.0, 0.0, 0.0, 0.0
  jdict, tdict, stats, AP, AP_class = [], [], [], [], []
  coco91class = coco80_to_coco91_class()

  #if args.cuda:
  #  fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  _t = {'im_detect': time.time(), 'misc': time.time()}
  #fasterRCNN.eval()

  #all_boxes = pickle.load(open(args.result_file, 'rb'))
  imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
  d_imgId_toIndex = {imid:i for i,imid in enumerate(imgIds)}

  if args.eval_file is None:
    cocoGt = COCO('data/coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
    cocoDt = cocoGt.loadRes(args.res_file)  # initialize COCO pred api

    resEv = COCOeval(cocoGt, cocoDt, 'bbox')
    resEv.params.imgIds = imgIds  # [:32]  # only evaluate these images
    resEv.evaluate()
    resEv.accumulate()
    resEv.summarize()
    import ipdb; ipdb.set_trace()
  else:
    resEv = pickle.load(open(args.eval_file,'rb'))
  print('Loaded evaluation results')
  img_save_dir = os.path.join('removal_results', 'yolov3', args.plot_type)
  if not os.path.isdir(img_save_dir):
    os.makedirs(img_save_dir)

  catToIndex = {cid:i for i, cid in enumerate(resEv.params.catIds)}
  imidToIndex = {imid:i for i, imid in enumerate(resEv.params.imgIds)}
  annIdToMaxDetectBox = defaultdict(list)
  imidToDetectBox = defaultdict(set)
  beforeAfters = []
  pltIds = []
  pltAnnIds= []
  pltCats = []

  curr_categ = resEv.cocoGt.getCatIds(args.class_name)[0]
  cat_index = resEv.params.catIds.index(curr_categ)
  curr_area = resEv.params.areaRngLbl.index(args.area) # Full range
  print(curr_area)

  r_start = cat_index * len(resEv.params.areaRng) * len(resEv.params.imgIds) + curr_area *  len(resEv.params.imgIds)
  r_end = r_start + len(resEv.params.imgIds)
  print(r_start,r_end)
  if args.plot_type == 'aggregate':
    frcnnImidToInd = {resEv.evalImgs[i]['image_id']:i for i in range(r_start, r_end) if (resEv.evalImgs[i] is not None)}
  else:
    frcnnImidToInd = {resEv.evalImgs[i]['image_id']:i for i in range(r_start, r_end) if (resEv.evalImgs[i] is not None) and (len(resEv.evalImgs[i]['gtIds'])>sum(resEv.evalImgs[i]['gtIgnore']))}
  if args.plot_type == 'spatial':
    annLocToMaxScore = []
    # This is for person only!
    for imid in frcnnImidToInd:
        ind = frcnnImidToInd[imid]
        gtMatches = resEv.evalImgs[ind]['gtMatches']
        img = resEv.cocoGt.loadImgs([resEv.evalImgs[ind]['image_id']])[0]
        imSz = [img['width'], img['height']]
        gtAnns = resEv.cocoGt.loadAnns(resEv.evalImgs[ind]['gtIds'])
        for i,annId in enumerate(resEv.evalImgs[ind]['gtIds']):
            if not resEv.evalImgs[ind]['gtIgnore'][i]:
                gtBox = gtAnns[i]['bbox']
                matchLoc = gtMatches[:,i].argmin()
                maxIou = resEv.params.iouThrs[matchLoc-1] if matchLoc > 0 else 0.
                matchDtId = gtMatches[matchLoc-1,i]
                maxMatchScore = resEv.evalImgs[ind]['dtScores'][resEv.evalImgs[ind]['dtIds'].index(matchDtId)] if matchLoc > 0 else 0.
                annLocToMaxScore.append([(gtBox[0]+gtBox[2]/2)/imSz[0],1.-(gtBox[1]+gtBox[3])/imSz[1], maxMatchScore, imid, curr_categ, annId])
                if matchLoc > 0:
                    annIdToMaxDetectBox[annId].append(matchDtId)

    annLocToMaxScore = np.array(annLocToMaxScore)
    pltColors = annLocToMaxScore[:,2]
    beforeAfters = annLocToMaxScore[:,:2]
    pltIds = annLocToMaxScore[:,3].astype(np.int)
    pltCats = annLocToMaxScore[:,-2].astype(np.int)
    pltAnnIds= annLocToMaxScore[:,-1].astype(np.int)
    print(len(annLocToMaxScore), len(frcnnImidToInd))
  elif args.plot_type == 'context':
    annNNdistToMaxScore = []
    annNNclsToMaxScore = []
    for imid in frcnnImidToInd:
        ind = frcnnImidToInd[imid]
        gtMatches = resEv.evalImgs[ind]['gtMatches']
        img = resEv.cocoGt.loadImgs([resEv.evalImgs[ind]['image_id']])[0]
        imSz = [img['width'], img['height']]
        gtAnns = resEv.cocoGt.loadAnns(resEv.evalImgs[ind]['gtIds'])
        allAnns = resEv.cocoGt.loadAnns(resEv.cocoGt.getAnnIds([imid]))
        dist = np.zeros((len(gtAnns),len(allAnns))) + 1.
        distCls = np.zeros((len(gtAnns),len(allAnns))) + 100
        for i,gA in enumerate(gtAnns):
            for j,aa in enumerate(allAnns):
                if gA['id'] != aa['id']:
                    dist[i,j] = l2box(gA['bbox'],aa['bbox'], imSz)
                    distCls[i,j] = aa['category_id']
        for i,annId in enumerate(resEv.evalImgs[ind]['gtIds']):
            if not resEv.evalImgs[ind]['gtIgnore'][i]:
                matchLoc = gtMatches[:,i].argmin()
                maxIou = resEv.params.iouThrs[matchLoc-1] if matchLoc > 0 else 0.
                matchDtId = gtMatches[matchLoc-1,i]
                maxMatchScore = resEv.evalImgs[ind]['dtScores'][resEv.evalImgs[ind]['dtIds'].index(matchDtId)] if matchLoc > 0 else 0.
                annNNdistToMaxScore.append([dist[i,:].min(), maxMatchScore, imid, curr_categ, annId])
                annNNclsToMaxScore.append([distCls[i,dist[i,:].argmin()], maxMatchScore, imid, curr_categ, annId])
                if matchLoc > 0:
                    annIdToMaxDetectBox[annId].append(matchDtId)
        imidToDetectBox[imid].update(resEv.cocoDt.getAnnIds(imid))

    annNNdistToMaxScore= np.array(annNNdistToMaxScore)
    annNNclsToMaxScore= np.array(annNNclsToMaxScore)
    pltColors = annNNdistToMaxScore[:,1]
    beforeAfters = np.concatenate([annNNclsToMaxScore[:,:1], annNNdistToMaxScore[:,:1]],axis=1)
    pltIds = annNNdistToMaxScore[:,2].astype(np.int)
    pltCats = annNNdistToMaxScore[:,-2].astype(np.int)
    pltAnnIds= annNNdistToMaxScore[:,-1].astype(np.int)
  elif args.plot_type == 'aggregate':
    dtScoreVsIou = []
    print(len(frcnnImidToInd))
    for imid in frcnnImidToInd:
        imidToDetectBox[imid].update(resEv.cocoDt.getAnnIds(imid))
        ind = frcnnImidToInd[imid]
        dtMatches = resEv.evalImgs[ind]['dtMatches']
        img = resEv.cocoGt.loadImgs([resEv.evalImgs[ind]['image_id']])[0]
        imSz = [img['width'], img['height']]
        gtAnns = resEv.cocoGt.loadAnns(resEv.evalImgs[ind]['gtIds'])
        dtAnns = resEv.cocoDt.loadAnns(resEv.evalImgs[ind]['dtIds'])
        gtSizes = [gt['area'] for gt in gtAnns]
        dtSizes = [dt['area'] for dt in dtAnns]
        matchedGts = []
        for i,annId in enumerate(resEv.evalImgs[ind]['dtIds']):
            if not resEv.evalImgs[ind]['dtIgnore'][:,i].all():
                matchLoc = dtMatches[:,i].argmin() if dtMatches[:,i].sum() > 0 else -1
                maxIou = resEv.params.iouThrs[matchLoc-1] if matchLoc >= 0 else 0.
                matchedGts += [dtMatches[matchLoc-1,i]] if matchLoc >=0 else []
                dtScoreVsIou.append([dtSizes[i], resEv.evalImgs[ind]['dtScores'][i], 2 if matchLoc>=0 else 1, imid, curr_categ, annId])
        matchedGts = set(matchedGts)
        for i,annId in enumerate(resEv.evalImgs[ind]['gtIds']):
            if annId not in matchedGts:
                dtScoreVsIou.append([gtSizes[i], -0.3, 0, imid, curr_categ, annId])


    dtScoreVsIou = np.array(dtScoreVsIou)
    beforeAfters = np.concatenate([dtScoreVsIou[:,:1], dtScoreVsIou[:,1:2]],axis=1)
    pltIds = dtScoreVsIou[:,-3].astype(np.int)
    pltCats = dtScoreVsIou[:,-2].astype(np.int)
    pltAnnIds= dtScoreVsIou[:,-1].astype(np.int)
    pltColors = dtScoreVsIou[:,2].astype(np.int)
  else:
      print('Unknown plot type')
      assert(0)


  def onpick(event):
      ind = event.ind
      imgId = pltIds[ind[0]]
      annId = pltAnnIds[ind[0]]
      oImgId = imgId#.split('_')[0]
      catId = pltCats[ind[0]]
      catname = resEv.cocoGt.cats[catId]['name']
      #removedcat = resEv.cocoGt.cats[int(imgId.split('_')[1])]['name']
      print('Image id %s, cat: %s'%(imgId, catname))

      # Load and prep the edited imageA
      #dImgInd = d_imgId_toIndex[imgId]
      #if len(bimidToMaxDetectBox[imgId,catId]):
      #  remBoxData = resEv.cocoDt.loadAnns(bimidToMaxDetectBox[imgId,catId])
      #  remBox = np.concatenate([np.array(ann['bbox'] + [ann['score']])[None,:] for ann in remBoxData],axis=0)
      #  remBox[:,2] = remBox[:,0] + remBox[:,2]
      #  remBox[:,3] = remBox[:,1] + remBox[:,3]
      #else:
      #  remBox = []

      #remImg = loadAndPrepImg(dataset[dImgInd], remBox, catname)

      # Load and prep the edited image
      #import ipdb; ipdb.set_trace()
      oImgInd = d_imgId_toIndex[oImgId]
      if args.plot_type == 'aggregate':
        annType = pltColors[ind[0]]
        if annType > 0:
            oBoxData = resEv.cocoDt.loadAnns([annId] + ([] if not args.showall else list(imidToDetectBox[oImgId] - set([annId]))))
        else:
            oBoxData = resEv.cocoGt.loadAnns([annId]) + resEv.cocoDt.loadAnns(imidToDetectBox[oImgId])
      else:
        oBoxData = resEv.cocoGt.loadAnns([annId])+resEv.cocoDt.loadAnns(annIdToMaxDetectBox[annId] + ([] if not args.showall else list(imidToDetectBox[oImgId] - set(annIdToMaxDetectBox[annId]))))
      if len(oBoxData):
          oBox = np.concatenate([np.array(ann['bbox'] + [ann['score'] if 'score' in ann else 1.1])[None,:] for ann in oBoxData],axis=0)
          oBox[:,2] = oBox[:,0] + oBox[:,2]
          oBox[:,3] = oBox[:,1] + oBox[:,3]
          catNames = [resEv.cocoGt.cats[ann['category_id']]['name'] for ann in oBoxData]
      else:
          oBox = []
          catNames = []
      oImg = loadAndPrepImg(dataset[oImgInd], oBox, catNames, th=args.dtTh)

      # Load and prep the edited image

      #cv2.imwrite('result.png', im2show)
      #pdb.set_trace()
      #concatImg = np.concatenate([oImg,remImg],axis=1).astype(np.uint8)
      concatImg = oImg
      cv2.imshow('concat',  concatImg)
      #cv2.imshow('original', oImg.astype(np.uint8))
      keyInp = cv2.waitKey(0)
      cv2.destroyWindow('concat')
      if (keyInp & 0xFF == ord('s')):
          if args.savesepimgs:
            saveIndividImages([oImg,remImg],['orignal','edited'], img_save_dir, oImgId, removedcat+'_'+catname)
          else:
            cv2.imwrite(os.path.join(img_save_dir,str(oImgId)+'_'+catname+'.png'), concatImg)
      return


  fig = plt.figure();ax = plt.subplot();
  fig.canvas.mpl_connect('pick_event', onpick)
  beforeAfters = np.array(beforeAfters)
  #cax = ax.scatter(np.log(beforeAfters[:,0]+1e-3), np.log(beforeAfters[:,1]+1e-3), alpha=0.5,c=pltCats,cmap=plt.cm.plasma,s=14, picker=True)
  cax = ax.scatter(beforeAfters[:,0], beforeAfters[:,1], alpha=0.4,c=pltColors,cmap=plt.cm.brg,s=14, picker=True)
  #ax.plot(np.arange(np.log(beforeAfters[:,0]+1e-3).min(), np.log(beforeAfters[:,0]+1e-3).max()),np.arange(np.log(beforeAfters[:,0]+1e-3).min(), np.log(beforeAfters[:,0]+1e-3).max()), 'k-',linewidth=2)

  if args.plot_type == 'spatial':
    ax.set_xlabel('Normalized X location of the box', fontsize=14);
    ax.set_ylabel('Normalized Y location of the box',fontsize=14);
    fig.colorbar(cax);
  elif args.plot_type == 'context':
    ax.set_xlabel('Class of the nearest box', fontsize=14);
    ax.set_ylabel('distance to the nearest box',fontsize=14);
    fig.colorbar(cax);
  elif args.plot_type == 'aggregate':
    ax.set_xscale('log')
    ax.set_xlabel('Area of the box', fontsize=14);
    ax.set_ylabel('Confidence score of the model',fontsize=14);
    fig.colorbar(cax);
  plt.show()



  end = time.time()
  print("test time: %0.4fs" % (end - start))
