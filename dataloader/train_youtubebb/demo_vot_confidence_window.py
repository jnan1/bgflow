# vot2018
#total entropy mean: 0.0724658410441471
#total scale mean: [1.0945092 0.7664999]
#total entropy std = 2.491323154407462
#total scale std = [6.43220842 3.79211775]
import os
import sys
import pickle
import numpy as np
from scipy.misc import imsave
sys.path.append('/home/jianingq/research_tool/visualization/')
sys.path.append('/home/jianingq/research_tool/datasets/vot/')
sys.path.append('/home/jianingq/research_tool/datasets/')
sys.path.append('/home/jianingq/PWC-Net/PyTorch')
sys.path.append('/home/jianingq/paper/pysot-toolkit')
from normal_loader import VOT
from save import save_sequences
from data_format import bbox_format
from image_processing import flow_warp, flow_warp_backward, add_mask, draw_bbox
from script_pwc import calculate_flow
import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join
from scipy import ndimage
from skimage import color

from net import SiamRPNvot
from .run_SiamRPN_confidence_window import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect, rect_2_cxy_wh
import time
from flowlib import read_flow
from pysot.utils.region import vot_overlap
#for compute-0-9
#root = '/scratch/jianren/Workspace/vot/SiamRPN'
#for compute-0-7
#root = '/scratch/jianren/vot_data'
#for compute-0-11
root = '/home/jianingq/vot-toolkit/SiamRPN/bgflow/sequences'
flow_dirs = '/scratch/jianren/backward_flows'
vot_dir = '/scratch/jianren/vot_data/'
def color_confidence(im1, im2): 
    #switch from BGR to RGB
    im1RGB = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
    im2RGB = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    im1LAB =  color.rgb2lab(im1RGB)
    im2LAB =  color.rgb2lab(im2RGB)
    diff = color.deltaE_ciede2000(im1LAB,im2LAB)
    diff = diff + 0.4
    return diff

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

def calculate_iou(predict_bbox, gtbbox):

    pred_polygon = (predict_bbox[0], predict_bbox[1], predict_bbox[0] + predict_bbox[2], predict_bbox[1],
                    predict_bbox[0] + predict_bbox[2], predict_bbox[1] + predict_bbox[3],predict_bbox[0],
                    predict_bbox[1] + predict_bbox[3])
    gt = gtbbox
    if(len(gt) == 4):
        gt_polygon = (gt[0], gt[1], 
                      gt[0] + gt[2], gt[1],
                      gt[0] + gt[2], gt[1] + gt[3], 
                      gt[0], gt[1] + gt[3])
    else:
        gt_polygon = (gt[0], gt[1], gt[2], gt[3], gt[4], gt[5], gt[6], gt[7])
    iou = vot_overlap(gt_polygon, pred_polygon,(im.shape[1], im.shape[0]))
    return iou

# cx: x for center of bounding box
# width: width of bounding box
def collision_detection(cx, x, width):
    dx = max(abs(cx - x) - width / 2, 0);
    return dx

def apply_mask(image,mask):
    mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
    mask = np.stack((mask,)*3, axis=-1)
    im_mask = add_mask(image, mask)
    return im_mask

vot = VOT(root)
video_names = vot.get_video_names()
# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'model','SiamRPNVOT.model')))
net.eval().cuda()

index_3 = 2.0
total_entropy_mean =0.0724658410441471
total_scale_mean =[1.0945092, 0.7664999]
total_entropy_std = 2.491323154407462
total_scale_std = [6.43220842, 3.79211775]

for video_name in ['soccer1']:
    with open('./confidence_window/'+video_name+'.txt', 'a') as f:
        for window_influence in [0.45,0.5,0.55,0.6,0.65,0.7,0.75]:
            for uncertainty_weight in [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,7]:
                total_iou = 0
                total_failure = 0

                warped_images = []
                video_length = vot.get_frame_length(video_name)
                gts = vot.get_gts(video_name)
                frame_tags = vot.get_frame_tags(video_name)
                video_frames = vot.get_frames(video_name)
                flow_dir = os.path.join(flow_dirs, video_name + '.txt')
                img_dir = os.path.join(vot_dir, video_name,'color')
                confidence_dir = os.path.join('/scratch/jianingq/backward_flow_confidence_vot/',video_name)

                #initialize network
                # image and init box
                init_rbox = gts[0]
                if(len(init_rbox) == 4):
                    [cx, cy], [w, h] = rect_2_cxy_wh(init_rbox)
                else:
                    [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

                # tracker init
                target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                im = np.copy(video_frames[0])# HxWxC
                state = SiamRPN_init(im, target_pos, target_sz, net)

                detection_box = state['detection_box']
                failure_counter = 0
                skip_loop = 0
                warped_images = []
                for i in range(0,video_length-1):
                    if(skip_loop==0):
                        #track
                        start = time.time()
                        im1 = np.copy(video_frames[i])
                        im = np.copy(video_frames[i + 1])
                        entropy_data = np.load(os.path.join(confidence_dir,format(i+1, '08')+'_entropy.npy'))
                        flow = np.load(os.path.join(confidence_dir,format(i+1, '08')+'_flow.npy'))
                        scale_data = np.load(os.path.join(confidence_dir,format(i+1, '08')+'_scale.npy'))
                        
                        ############################################

                        current_bbox = bbox_format(gts[i],'tlxy_wh_2_rect')
                        current_bbox = [int(m) for m in current_bbox ]

                        #cv2.rectangle(im, (current_bbox[0], current_bbox[1]), (current_bbox[2], current_bbox[3]), (0, 0, 0), 2)
                        #x0,y0,x1,y1
                        next_bbox = bbox_format(gts[i + 1],'tlxy_wh_2_rect')
                        next_bbox = [int(l) for l in next_bbox]

                        next_mask = background_flow(im, detection_box, flow)

                        ##################confidence window##############
                        im_height, im_width, _ = np.shape(im) 
                        box_width =  (detection_box[2] - detection_box[0])
                        box_height = (detection_box[3] - detection_box[1])
                        cx = (detection_box[0] + detection_box[2])/2
                        cy = (detection_box[1] + detection_box[3])/2
                        xx,yy =  np.meshgrid([x for x in range(im_width)],
                                             [y for y in range(im_height)])
                        xx = xx + flow[:,:,0]
                        yy = yy + flow[:,:,1]
                        collision_detection_vec = np.vectorize(collision_detection)
                        xx = collision_detection_vec(cx, xx, box_width)
                        yy = collision_detection_vec(cy, yy, box_height)
                        #xx,yy = np.meshgrid([collision_detection(cx,i,box_width) for i in range(im_width)],
                        #                    [collision_detection(cy,j,box_height) for j in range(im_height)])
                        distance_mask = np.sqrt(xx ** 2 + yy ** 2)
                        scale_dx = (scale_data[:,:,0]-total_scale_mean[0])/total_scale_std[0]
                        scale_dy = (scale_data[:,:,1]-total_scale_mean[1])/total_scale_std[1]
                        confidence_window = np.exp(-(np.sqrt(xx**2)/(uncertainty_weight*((scale_dx+0.0001)**2))) - np.sqrt(yy**2)/(uncertainty_weight*((scale_dy+0.0001)**2)))
                        ###################################################################

                        state = SiamRPN_track(state, im, confidence_window, window_influence, gts[i+1])
                        #x0,y0,w,h
                        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                        temp_result_box = res
                        #x0,y0,x1,y1
                        detection_box = [int(res[0]), int(res[1]), int(res[0] + res[2]), int(res[1] + res[3])]

                        cv2.rectangle(im, (detection_box[0], detection_box[1]), (detection_box[2], detection_box[3]), (0, 255, 0), 2)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(im,str(i),(20,230), font, 0.6,(255,255,255), 1,cv2.LINE_AA)

                        iou = calculate_iou(res,gts[i+1])
                        total_iou = total_iou + iou

                        #enlarge the detection box 

                        detection_box = [np.clip(int(detection_box[0]), 0, im.shape[1]-1),\
                                         np.clip(int(detection_box[1]), 0, im.shape[0]-1),\
                                         np.clip(int(detection_box[2]), 0, im.shape[1]-1),\
                                         np.clip( int(detection_box[3]), 0, im.shape[0]-1)]
                        
                        distance_image = apply_mask(im,distance_mask)

                        confidence_mask = apply_mask(im,confidence_window+0.4)

                        entropy_mask = apply_mask(im,entropy_data[:,:,0])

                        im = draw_bbox(im, next_bbox, (255, 255, 255), 4)


                        vis = np.concatenate((im, entropy_mask), axis=1)
                        vis2 = np.concatenate((distance_image, confidence_mask), axis=1)
                        final_im = np.concatenate((vis,vis2), axis=0)

                        warped_images.append(final_im)

                        if(iou <= 0):
                            failure_counter = failure_counter+1
                            total_failure = total_failure + 1
                            skip_loop = 1
                    else:
                        failure_counter = failure_counter+1
                        next_bbox = bbox_format(gts[i + 1],'tlxy_wh_2_rect')
                        next_bbox = [int(l) for l in next_bbox]
                        im = video_frames[i+1]
                        if(failure_counter == 6):
                            #tracker init
                            if(len(gts[i + 1]) == 4):
                                [cx, cy], [w, h] = rect_2_cxy_wh(gts[i + 1])
                            else:
                                [cx, cy, w, h] = get_axis_aligned_bbox(gts[i + 1])
                            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                            state = SiamRPN_init(im, target_pos, target_sz, net)
                            detection_box = state['detection_box']
                            failure_counter = 0
                            skip_loop = 0
                        image_mask = draw_bbox(im, next_bbox, (255, 255, 255), 4)
                        vis = np.concatenate((im, im), axis=1)
                        vis2 = np.concatenate((im, image_mask), axis=1)
                        final_im = np.concatenate((vis,vis2), axis=0)
                        warped_images.append(final_im)
                #save_sequences(warped_images, export_dir='./confidence_window/'+video_name+'_test_'+str(window_influence)+'.mp4')
                precision = float(total_iou)/float(video_length)
                print("window_influence: %.2f, uncertainty_weight: %.2f "%(window_influence,uncertainty_weight))
                f.write("window_influence, uncertainty_weight: " + str(window_influence) + str(uncertainty_weight) + '\n')
                print("total_failure : " + str(total_failure))
                f.write("total_failure : " + str(total_failure) + '\n')
                print("precision: " + str(precision))
                f.write("precision: " + str(precision) + '\n')
                f.flush()
    f.close()
                #################################################





