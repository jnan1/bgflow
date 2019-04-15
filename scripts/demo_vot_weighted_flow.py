import os
import sys
import pickle
import numpy as np
from scipy.misc import imsave
sys.path.append('/home/jianingq/research_tool/visualization/')
sys.path.append('/home/jianingq/research_tool/datasets/vot/')
sys.path.append('/home/jianingq/research_tool/datasets/')
sys.path.append('/home/jianingq/PWC-Net/PyTorch')
from normal_loader import VOT
from save import save_sequences
from data_format import bbox_format
from image_processing import flow_warp, flow_warp_backward, add_mask, draw_bbox
from script_pwc import calculate_flow
import glob, cv2, torch
import numpy as np
from os.path import realpath, dirname, join
from scipy import ndimage
from scipy.spatial import distance
from skimage import color

from net import SiamRPNvot
from run_SiamRPN_network import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect, rect_2_cxy_wh
import time

#for compute-0-9
#root = '/scratch/jianren/Workspace/vot/SiamRPN'
#for compute-0-7
#root = '/scratch/jianren/vot_data'
root = '/home/jianingq/vot-toolkit/SiamRPN/bgflow/sequences'
flow_dirs = '/scratch/jianren/backward_flows'
#vot_dir = '/scratch/jianren/vot_data/'
vot_dir = '/scratch/jianingq/vot_data/'

#im1 im2 both by cv2.imread
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

def calculate_iou(bbox1, bbox2):

    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = intersection_area / \
        float(bbox1_area + bbox2_area - intersection_area)
    return iou

vot = VOT(root)
video_names = vot.get_video_names()
# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'model','SiamRPNVOT.model')))
net.eval().cuda()


for video_name in ['flamingo1']:#['drone1','gymnastics3','handball2','soccer1','condunction1']:#['crabs1','gymnastics2','gymnastics3','book','bolt1','bolt2','hand','basketball','iceskater2']:s
    for index_1 in [0.3]:#np.linspace(0.3,1,8):
        for index_2 in [1.3]:#np.linspace(0,5,51):
            for index_3 in [1]:# np.linspace(0.5,2,4):
                total_iou = 0
                total_failure = 0
                #video_name = 'gymnastics3'
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
                im2 = video_frames[0]# HxWxC
                detection_box = [int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)]
                foreground_candidate_mask = np.zeros(im2.shape) 
                foreground_candidate_mask[detection_box[1]:detection_box[3], detection_box[0]:detection_box[2], :] = 1
                state = SiamRPN_init(im2, target_pos, target_sz, net)
                font = cv2.FONT_HERSHEY_SIMPLEX
                data_dir = os.path.join('/home/jianingq/vot_info',video_name)
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir)
                for i in range(0,video_length-1):
                    im1 = np.copy(video_frames[i])
                    im2 = np.copy(video_frames[i + 1])
                    current_bbox = bbox_format(gts[i],'tlxy_wh_2_rect')
                    current_bbox = [int(j) for j in current_bbox]
                    next_bbox = bbox_format(gts[i+1],'tlxy_wh_2_rect')
                    next_bbox = [int(j) for j in next_bbox]
                    flow = np.load(os.path.join(confidence_dir,format(i+1, '08')+'_flow.npy'))
                    entropy_data = np.load(os.path.join(confidence_dir,format(i+1, '08')+'_entropy.npy'))
                    entropy = (entropy_data - np.min(entropy_data))/(np.max(entropy_data)-np.min(entropy_data))
                    confidence = 1-entropy
                    warped_im1 = flow_warp(im1, flow)
                    diff, rgb_diff = color_confidence(warped_im1, im2)

                    #next_mask = background_flow(im2, detection_box, flow)
                    next_mask = flow_warp_backward(foreground_candidate_mask, flow)
                    # np.save(os.path.join(data_dir,format(i+1, '08')+"_fgmask.npy"), next_mask)
                    if(i > 0):
                        pre_flow_mask = np.load(os.path.join(data_dir,format(i, '08')+"_fgmask.npy"))
                        double_warped = flow_warp(pre_flow_mask, flow)
                        # np.save(os.path.join(data_dir,format(i+1, '08')+"_prev_fgmask.npy"), double_warped)

                    overall_mask = (confidence[:,:,0]*next_mask[:,:,0] + entropy[:,:,0])/2

                    state = SiamRPN_track(state, im2, overall_mask, confidence[:,:,0], next_mask[:,:,0], entropy[:,:,0], index_1,index_2,gts[i+1])

                    foreground_candidate_mask = state['fg']

                    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                    ##print(res)
                    #x0,y0,w,h
                    res = [int(l) for l in res]
                    #x0,y0,x1,y1
                    detection_box = [res[0], res[1], res[0] + res[2], res[1] + res[3]]

                    cv2.rectangle(im2,(next_bbox[0], next_bbox[1]), (next_bbox[2], next_bbox[3]), (255,255, 255), 3)
                    cv2.rectangle(im2, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0,0 , 255), 2)

                    diff = diff/(np.max(diff)-np.min(diff))

                    diff_mask = np.zeros((np.shape(diff)[0],np.shape(diff)[1],3),dtype='float')
                    diff_mask[:,:,0] = diff
                    diff_mask[:,:,1] = diff
                    diff_mask[:,:,2] = diff
                    # np.save(os.path.join(data_dir,format(i+1, '08')+"_lab.npy"), diff_mask)

                    entropy_mask = np.zeros((np.shape(entropy)[0],np.shape(entropy)[1],3),dtype='float')
                    entropy_mask[:,:,0] = entropy[:,:,0]
                    entropy_mask[:,:,1] = entropy[:,:,0]
                    entropy_mask[:,:,2] = entropy[:,:,0]
                    # np.save(os.path.join(data_dir,format(i+1, '08')+"_entropy.npy"), entropy_mask)

                    confidence_mask = np.zeros((np.shape(entropy)[0],np.shape(entropy)[1],3),dtype='float')
                    confidence_mask[:,:,0] = confidence[:,:,0]
                    confidence_mask[:,:,1] = confidence[:,:,0]
                    confidence_mask[:,:,2] = confidence[:,:,0]
                    # np.save(os.path.join(data_dir,format(i+1, '08')+"_confidence.npy"), confidence_mask)

                    overall_mask = (next_mask + entropy_mask)/2


                    #visualization
                    im1_mask = add_mask(im1,foreground_candidate_mask)
                    image_mask = add_mask(im2, overall_mask)
                    image_mask_entropy = add_mask(im2, entropy_mask)
                    image_mask_confidence = add_mask(im2, confidence_mask)
                    image_mask_flow = add_mask(im2,next_mask)

                    vis = np.concatenate((im1_mask,image_mask), axis=1)
                    vis2 = np.concatenate((image_mask_confidence, image_mask_flow), axis=1)
                    final_im = np.concatenate((vis,vis2), axis=0)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(image_mask,str(i),(20,230), font, 0.6,(255,255,255), 1,cv2.LINE_AA)

                    #restart if iou is too low
                    iou = calculate_iou(bbox_format(res,'tlxy_wh_2_rect'), bbox_format(gts[i + 1],'tlxy_wh_2_rect'))
                    total_iou = total_iou + iou
                    #enlarge the detection box

                    detection_box = [np.clip(int(detection_box[0]-0.1*res[2]), 0, im2.shape[1]-1),\
                                     np.clip(int(detection_box[1] - 0.1*res[3]), 0, im2.shape[0]-1),\
                                     np.clip(int(detection_box[2]+0.1*res[2]), 0, im2.shape[1]-1),\
                                     np.clip( int(detection_box[3]+0.1*res[3]), 0, im2.shape[0]-1)]

                    if(iou <= 0):
                        total_failure = total_failure + 1
                        cv2.rectangle(image_mask, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 0, 255), 3)
                        #tracker init
                        if(len(gts[i + 1]) == 4):
                            [cx, cy], [w, h] = rect_2_cxy_wh(gts[i + 1])
                        else:
                            [cx, cy, w, h] = get_axis_aligned_bbox(gts[i + 1])
                        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                        detection_box = [int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)]
                        im2 = video_frames[i+1]
                        state = SiamRPN_init(im2, target_pos, target_sz, net)
                    warped_images.append(final_im)
                save_sequences(warped_images, export_dir='./rerank_flow_2/weighted_flow/'+video_name+'_confidence_without_everything.mp4')


    #f.close()

