import os
import sys
import pickle
import numpy as np
from scipy.misc import imsave
sys.path.append('/home/jianingq/research_tool/visualization/')
sys.path.append('/home/jianingq/research_tool/datasets/vot/')
sys.path.append('/home/jianingq/research_tool/datasets/')
#Must import tf before torch!!!
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
import torch
from save import save_sequences
from data_format import bbox_format
from image_processing import flow_warp_backward, add_mask, draw_bbox
import glob, cv2
import numpy as np
from os.path import realpath, dirname, join
from scipy import ndimage

from .DaSiamRPN_net import SiamRPNvot
from .run_SiamRPN_youtubebb import SiamRPN_init, SiamRPN_track
import time
#sys.path.append('/home/jianingq/netdef_models/FlowNetH/Pred-Merged-SS')
#sys.path.append('/home/jianingq/lmbspecialops/python')
#import netdef_slim as nd
#import controller
#from netdef_slim.utils import io
#path = os.getcwd()
from skimage import color
#for compute-0-9
#root = '/scratch/jianren/Workspace/vot/SiamRPN'
#for compute-0-7
#root = '/scratch/jianren/vot_data'
root = '/home/jianingq/vot-toolkit/SiamRPN/bgflow/sequences'
flow_dirs = '/scratch/jianren/backward_flows'
vot_dir = '/scratch/jianren/vot_data/'
data_dir = '/scratch/jianingq/youtube-bb/yt_bb_detection_train'
from scipy import misc

def rect_2_cxy_wh(rect):
    return np.array([rect[0]+rect[2]/2, rect[1]+rect[3]/2]), np.array([rect[2], rect[3]])  # 0-index

def nrmse(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    a, b, c = im1.shape
    rmse = np.sqrt(np.sum(np.sum((im2 - im1) ** 2))/ float(a * b))
    max_val = max(np.max(im1), np.max(im2))
    min_val = min(np.min(im1), np.min(im2))
    return 1 - (rmse / (max_val - min_val))

def background_flow(next_image, current_bbox, backward_flow):
    mask = np.zeros(next_image.shape) + 0.3
    mask[current_bbox[1]:current_bbox[3], current_bbox[0]:current_bbox[2], :] = 1
    next_mask = flow_warp_backward(mask, backward_flow)

    return next_mask

#current_bbox: [x0, y0, x1, y2]
def forward_warp(frame2, current_bbox, flow):
    current_bbox[0] = np.clip(current_bbox[0], 0, frame2.shape[1]-1)
    current_bbox[1] = np.clip(current_bbox[1], 0, frame2.shape[0]-1)
    current_bbox[2] = np.clip(current_bbox[2], 0, frame2.shape[1]-1)
    current_bbox[3] = np.clip(current_bbox[3], 0, frame2.shape[0]-1)
    mask = np.zeros(frame2.shape).astype(int)
    for j in range(current_bbox[1],current_bbox[3]-1):
        for i in range(current_bbox[0],current_bbox[2]-1):
            dx = flow[j,i,0]
            dy = flow[j,i,1]
            x = np.floor(i + dx).astype(int)
            y = np.floor(j + dy).astype(int)
            x = np.clip(x, 0, frame2.shape[1]-1)
            y = np.clip(y, 0, frame2.shape[0]-1)
            mask[y,x] = 1
    kernel = np.ones((2,2),np.uint8)
    mask_erosion = cv2.erode((mask).astype(np.uint8),kernel)
    return mask_erosion

# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join('/home/jianingq/bgflow/DaSiamRPN/code/', 'model','SiamRPNVOT.model')))
net.eval().cuda()
#"dinosaur","gymnastics3",

def youtube_to_rec(box,im_h, im_w):
    template_x1 = int(box[0] * im_w)
    template_x2 = int(box[1] * im_w)
    template_y1 = int(box[2] * im_h)
    template_y2 = int(box[3] * im_h)
    w = template_x2-template_x1
    h = template_y2-template_y1
    return template_x1,template_y1,w,h

def color_confidence(im1, im2): 
    #switch from BGR to RGB
    im1RGB = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2RGB = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im1LAB =  color.rgb2lab(im1RGB)
    im2LAB =  color.rgb2lab(im2RGB)
    diff = color.deltaE_ciede2000(im1LAB,im2LAB)
    dist = (im1RGB[:,:,0]-im2RGB[:,:,0])**2+(im1RGB[:,:,1]-im2RGB[:,:,1])**2 + (im1RGB[:,:,2]-im2RGB[:,:,2])**2
    rgb_diff = dist
    return diff, rgb_diff

def generate_masks(index, train_im_paths, train_labels):
    im_paths = train_im_paths[index]
    labels = train_labels[index]

    flow_data = []
    entropy_data = []
    scale_data = []
    lab_data = []
    iou_data = []
    detection_data = []
    box_data = []

    warped_images = []
    box_data = []
    for i in range(0,3):
        im = np.copy(cv2.imread(im_paths[i]))
        im_h, im_w, im_c = im.shape
        names = im_paths[i].split('/')[-1].split('_')
        if(len(names)>4):
            video_name = ''
            for more_index in range(0,len(names)-3):
                if(more_index > 0):
                    video_name += '_'
                video_name += names[more_index] 
            frame_name = names[len(names)-3]
            type_name = names[len(names)-2]
        else:
            video_name = names[0]
            frame_name = names[1]
            type_name = names[2]
        class_dir = os.path.join(data_dir, type_name, video_name)
        #x0,y0,w,h
        gt = youtube_to_rec(labels[i],im_h,im_w)
        flow = np.zeros((im_h,im_w,2))
        if(i == 0):
            filename = os.path.join(class_dir,str(frame_name))
            with open(filename+'.txt') as ff:
                content = ff.readlines()
            ff.close()

        iou_result = []
        detection_score = []
        temp_result = []
        for j in range(0,100):
            data = content[100*i+j].split()
            x0,y0,w,h = data[0:4]
            res = [int(x0),int(y0),int(w),int(h)]
            temp_result.append(res)
            iou_result.append(float(data[4]))
            detection_score.append(float(data[5]))
        iou_data.append(iou_result)
        detection_data.append(detection_score)
        box_data.append(temp_result)

        
        if(i > 0):
            #im = cv2.imread(im_paths[i-1])
            flow = np.load(os.path.join(class_dir,str(frame_name)+'_flow.npy'))
            entropy = np.load(os.path.join(class_dir,str(frame_name)+'_entropy.npy'))
            warped_im = flow_warp_backward(im,flow)
            diff, rgb_diff = color_confidence(warped_im, im)
            flow_data.append(flow)
            lab_data.append(diff)
            entropy_data.append(entropy)
    
    return flow_data, entropy_data, lab_data, iou_data, detection_data, box_data

#flow_data: [2, image height, image width, 2]; flow_data[0] is flow from im2 to im1
#iou_result: [3,100] iou for 100 boxes for 3 images
#detection_data: [3,100] detection score for 100 boxes for 3 images
#box_data: [3,100,4] bounding box location for 100 boxes for 3 images, x0,y0,w,h

