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
from skimage import color
from run_SiamRPN_network import SiamRPN_init, SiamRPN_track_bbox
sys.path.append('/home/jianingq/bgflow/DaSiamRPN/code/')
from net import SiamRPNvot
from utils import get_axis_aligned_bbox, cxy_wh_2_rect, rect_2_cxy_wh
import time
from flowlib import read_flow
from score_weights import ScoreNet
#for compute-0-9
#root = '/scratch/jianren/Workspace/vot/SiamRPN'
#for compute-0-7
#root = '/scratch/jianren/vot_data'
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
"""
video_names = [
            'ants1', 'ants3', 'bag', 'ball1', 'ball2', 'basketball', 'birds1',
            'blanket', 'bmx', 'bolt1', 'bolt2', 'book', 'butterfly', 'car1',
            'conduction1', 'crabs1', 'crossing', 'dinosaur', 'drone_across',
            'drone_flip', 'drone1', 'fernando', 'fish1', 'fish2', 'fish3',
            'flamingo1', 'frisbee', 'girl', 'glove', 'godfather', 'graduate',
            'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'handball1',
            'handball2', 'helicopter', 'iceskater1', 'iceskater2', 'leaves',
            'matrix', 'motocross1', 'motocross2', 'nature', 'pedestrian1',
            'rabbit', 'racing', 'road', 'shaking', 'sheep', 'singer2',
            'singer3', 'soccer1', 'soccer2', 'soldier', 'tiger', 'traffic',
            'wiper', 'zebrafish1']
"""
vot = VOT(root)
video_names = vot.get_video_names()
# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join('/home/jianingq/bgflow/DaSiamRPN/code/', 'model','SiamRPNVOT.model')))
net.eval().cuda()

score_net = ScoreNet(6, 1)
score_net.cuda()
score_net.load_state_dict(torch.load('models/ckpt_0.pth'))

for video_name in ['gymnastics3']:

    total_iou = 0
    total_failure = 0
    warped_images = []
    video_length = vot.get_frame_length(video_name)
    #ground truth bounding box
    gts = vot.get_gts(video_name)
    frame_tags = vot.get_frame_tags(video_name)
    video_frames = vot.get_frames(video_name)
    flow_dir = os.path.join(flow_dirs, video_name + '.txt')
    img_dir = os.path.join(vot_dir, video_name,'color')
    confidence_dir = os.path.join('/home/jianingq/backward_flow_confidence_vot/',video_name)

    #initialize network
    # image and init box
    init_rbox = gts[0]
    if(len(init_rbox) == 4):
        [cx, cy], [w, h] = rect_2_cxy_wh(init_rbox)
    else:
        [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

    # tracker init
    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    im = video_frames[0]# HxWxC
    state = SiamRPN_init(im, target_pos, target_sz, net)

    detection_box = [int(cx-w/2),int(cy-h/2),int(cx+w/2),int(cy+h/2)]

    for i in range(0,video_length - 1):
        #track
        im1 = np.copy(video_frames[i])
        im2 = np.copy(video_frames[i + 1])
        entropy_data = np.load(os.path.join(confidence_dir,format(i+1, '08')+'_entropy.npy'))
        flow = np.load(os.path.join(confidence_dir,format(i+1, '08')+'_flow.npy'))
        warped_im1 = flow_warp(im1, flow)
        
        #############confidence map processing#######
        mean_entropy = entropy_data.flatten().mean()
        std_entropy = np.std(entropy_data.flatten(), ddof=1)
        entropy_bi_map = np.zeros(np.shape(entropy_data))

        #threshold entropy
        entropy_bi_map[entropy_data>(mean_entropy+1.0*std_entropy)] = 1
        entropy_bi_map[entropy_data<=(mean_entropy+1.0*std_entropy)] = 0

        #connected component
        blur_radius = 10.0
        ndimage.gaussian_filter(entropy_bi_map[:,:,0],blur_radius)
        labeled, nr_objects = ndimage.label(entropy_bi_map[:,:,0])
        binary_labeled = np.zeros(np.shape(labeled),dtype='bool')
        #x0 y0 x1 y1
        patch = labeled[detection_box[1]:detection_box[3],detection_box[0]:detection_box[2]]
        foreground_rank = (np.bincount(patch.flatten())).argsort()[-2:][::-1]
        count_occurance = np.bincount(labeled.flatten())
        foreground = foreground_rank[0]
        if(count_occurance[foreground] > 0.4*(np.size(im)/3)):
            if(len(foreground_rank) == 1):
                pass
            else:
                foreground = foreground_rank[1]
                foreground_index = np.where(((labeled == foreground)))
                binary_labeled[foreground_index] = 1
        else:
            #foreground = np.argmax(np.bincount(patch.flatten()))
            foreground_index = np.where(((labeled == foreground)))
            binary_labeled[foreground_index] = 1
        ##################combined#####################
        bi_mask = np.zeros((np.shape(binary_labeled)[0],np.shape(binary_labeled)[1],3),dtype='bool')
        bi_mask[:,:,0] = binary_labeled
        bi_mask[:,:,1] = binary_labeled
        bi_mask[:,:,2] = binary_labeled

        ############################################

        current_bbox = bbox_format(gts[i],'tlxy_wh_2_rect')
        next_bbox = bbox_format(gts[i + 1],'tlxy_wh_2_rect')

        next_bbox = [int(l) for l in next_bbox]

        next_mask = background_flow(im, detection_box, flow)

        state = SiamRPN_track_bbox(score_net, state, im, (next_mask[:,:,0]>0.99) ,(bi_mask[:,:,0]), gts[i + 1])

        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        #x0,y0,w,h
        res = [int(l) for l in res]
        #x0,y0,x1,y1
        detection_box = [res[0], res[1], res[0] + res[2], res[1] + res[3]]

        foreground_index =  np.bitwise_or(((next_mask[:,:,0])>0.99) ,bi_mask[:,:,0])
        foreground_mask = 0.4*np.ones(next_mask.shape)
        foreground_mask[foreground_index,:] = 1

        image_mask = add_mask(im, foreground_mask)
        image_mask = draw_bbox(image_mask, next_bbox, (255, 255, 255), 4)
        cv2.rectangle(image_mask, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_mask,str(i),(20,230), font, 0.6,(255,255,255), 1,cv2.LINE_AA)

        #restart if iou is too low
        iou = calculate_iou(bbox_format(res,'tlxy_wh_2_rect'), bbox_format(gts[i + 1],'tlxy_wh_2_rect'))
        total_iou = total_iou + iou
        #enlarge the detection box 

        detection_box = [np.clip(int(detection_box[0]-0.1*res[2]), 0, im.shape[1]-1),\
                         np.clip(int(detection_box[1] - 0.1*res[3]), 0, im.shape[0]-1),\
                         np.clip(int(detection_box[2]+0.1*res[2]), 0, im.shape[1]-1),\
                         np.clip( int(detection_box[3]+0.1*res[3]), 0, im.shape[0]-1)]

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
            im = video_frames[i+1]
            state = SiamRPN_init(im, target_pos, target_sz, net)

        warped_images.append(image_mask)

    precision = float(total_iou)/float(video_length)
    print(video_name)
    #f.write(video_name + '\n')
    print("total_failure : " + str(total_failure))
    #f.write("total_failure : " + str(total_failure) + '\n')
    print("precision: " + str(precision))
    #f.write("precision: " + str(precision) + '\n')
    #f.flush()
    #save_sequences(warped_images, export_dir='./rerank_flow_2/k1k2/'+video_name+'.mp4')
    #save_sequences(warped_images, export_dir='./rerank_flow_2/k1k2/'+video_name+'_confidence_'+str(index_1) + '_' + str(index_2) + '_' + str(index_3) + '.mp4')


