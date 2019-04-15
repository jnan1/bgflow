import os
import sys
import numpy as np
import pickle
sys.path.append('/home/jianingq/research_tool/visualization/')
sys.path.append('/home/jianingq/research_tool/datasets/vot/')
sys.path.append('/home/jianingq/research_tool/datasets/')
sys.path.append('/home/jianingq/lmbspecialops/python')

#from normal_loader_2016 import VOT2016
from normal_loader import VOT
from save import save_sequences
from data_format import bbox_format
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import torch
from image_processing import flow_warp_backward, add_mask, draw_bbox
import glob, cv2
from os.path import realpath, dirname, join
#for compute-0-9
#root = '/scratch/jianren/Workspace/vot/SiamRPN'
#for compute-0-7
root = '/scratch/jianren/vot_data'
vot_dir = '/scratch/jianren/vot_data'
#root = '/scratch/jianingq/VOT2016/'
#vot_dir = '/scratch/jianingq/VOT2016/'
#for compute-0-11
#vot_dir = '/scratch/jianingq/vot_data'
#root = '/scratch/jianingq/vot_data/'

sys.path.append('/home/jianingq/netdef_models/FlowNetH/Pred-Merged-SS')
sys.path.append('/home/jianingq/lmbspecialops/python')
import netdef_slim as nd
import controller
from netdef_slim.utils import io
path = os.getcwd()


def youtube_to_rec(box,im_h, im_w):
    im_h, im_w, im_c = im.shape
    template_x1 = int(box[0] * im_w)
    template_x2 = int(box[1] * im_w)
    template_y1 = int(box[2] * im_h)
    template_y2 = int(box[3] * im_h)
    w = template_x2-template_x1
    h = template_y2-template_y1
    return template_x1,template_y1,w,h
train_im_paths = pickle.load(open('./train_youtubebb_image_python3.txt','rb'))
train_labels = pickle.load(open('./train_youtubebb_labels_python3.txt','rb'))

print(np.shape(train_im_paths))
im_paths = train_im_paths[1000]

labels = train_labels[1000]

flow_data = []
entropy_data = []
scale_data = []

#initialize network
# image and init box
im = cv2.imread(im_paths[0])# HxWxC
im_h, im_w, im_c = im.shape
x0,y0,w,h = youtube_to_rec(labels[0],im_h,im_w)
init_rbox = [x0,y0,w,h]
#[cx, cy], [w, h] = rect_2_cxy_wh(init_rbox)
#target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
#state = SiamRPN_init(im, target_pos, target_sz, net)

# tracker init
#target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
cv2.rectangle(im, (x0,y0), (x0+w, y0+h), (0, 255, 0), 2)

warped_images = []


c = nd.load_module('/home/jianingq/netdef_models/FlowNetH/Pred-Merged-SS/controller.py').Controller()

for i in range(0,2):
    c_net = c.net_actions(net_dir='/home/jianingq/netdef_models/FlowNetH/Pred-Merged-SS')
    im1 = im_paths[i]
    im2 = im_paths[i+1]
    if i == 0:
        eval_out, session = c_net.init_eval(im2, im1)
    out = c_net.simple_eval(eval_out, session, im2, im1)

    for k,v in out.items():
        if(k =='flow[0][1].fwd'):
            flow_data.append(v[0,:,:,:].transpose(1,2,0))
        elif(k == 'iul_entropy[0][1].fwd'):
            entropy_data.append(v[0,:,:,:].transpose(1,2,0))
        if(k == 'iul_scale[0][1].fwd'):
            scale_data.append(v[0,:,:,:].transpose(1,2,0))



