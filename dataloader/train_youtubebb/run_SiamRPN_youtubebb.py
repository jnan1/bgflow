# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
import glob, cv2, torch
from torch.autograd import Variable
import torch.nn.functional as F
from .utils import cxy_wh_2_rect
from .utils import get_subwindow_tracking,rect_2_cxy_wh
from image_processing import flow_warp_backward, add_mask, draw_bbox
import time
import sys
sys.path.append('/home/jianingq/paper/pysot-toolkit')
from data_format import bbox_format
from pysot.utils.region import vot_overlap
sys.path.append('/home/jianingq/research_tool/visualization/')
sys.path.append('/home/jianingq/research_tool/datasets/vot/')
sys.path.append('/home/jianingq/research_tool/datasets/')
sys.path.append('/home/jianingq/PWC-Net/PyTorch')
sys.path.append('/home/jianingq/paper/pysot-toolkit')

np.set_printoptions(threshold=np.inf)
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

def generate_anchor(total_stride, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4),  dtype=np.float32)
    size = total_stride * total_stride
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1

    anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
    ori = - (score_size / 2) * total_stride
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
             np.tile(yy.flatten(), (anchor_num, 1)).flatten()
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor


class TrackerConfig(object):
    # These are the default hyper-params for DaSiamRPN 0.3827
    windowing = 'cosine'  # to penalize large displacements [cosine/uniform]
    # Params from the network architecture, have to be consistent with the training
    exemplar_size = 127  # input z size
    instance_size = 271  # input x size (search region)
    total_stride = 8
    score_size = (instance_size-exemplar_size)/total_stride+1
    context_amount = 0.5  # context amount for the exemplar
    ratios = [0.33, 0.5, 1, 2, 3]
    scales = [8, ]
    anchor_num = len(ratios) * len(scales)
    anchor = []
    penalty_k = 0.055
    window_influence = 0.42
    lr = 0.295
    # adaptive change search region #
    adaptive = True

    def update(self, cfg):
        for k, v in cfg.items():
            setattr(self, k, v)
        self.score_size = (self.instance_size - self.exemplar_size) / self.total_stride + 1



def tracker_eval_box(net, x_crop, target_pos, target_sz, window, scale_z, p, im, gtbbox):
    delta, score = net(x_crop)
    foreground_candidate_mask = np.zeros((np.shape(im)[0],np.shape(im)[1],3))

    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1./r)

    def sz(w, h):
        pad = (w + h) * 0.5
        sz2 = (w + pad) * (h + pad)
        return np.sqrt(sz2)

    def sz_wh(wh):
        pad = (wh[0] + wh[1]) * 0.5
        sz2 = (wh[0] + pad) * (wh[1] + pad)
        return np.sqrt(sz2)

    # size penalty
    s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
    r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

    penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
    pscore = penalty * score

    # window float
    pscore = pscore * (1 - p.window_influence) + window * p.window_influence
    target_sz = target_sz / scale_z

    inspect_num = 100
    top_score = score.argsort()[-inspect_num:][::-1]
    font = cv2.FONT_HERSHEY_SIMPLEX

    iou_result = np.zeros(inspect_num)
    detection_score = np.zeros(inspect_num)
    temp_result = np.zeros((inspect_num,4),dtype=int)
    score_result = np.zeros(inspect_num)
    for i in range(99,-1,-1):
        target = delta[:, top_score[i]] / scale_z
        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]

        res_w = target[2]
        res_h = target[3]
        res = cxy_wh_2_rect(np.array([res_x, res_y]), np.array([res_w, res_h]))

        temp = [res[0],res[1],(res[0] + res[2]),(res[1] + res[3])]
        res = [np.clip(temp[0], 0, im.shape[1]-1),\
               np.clip(temp[1], 0, im.shape[0]-1),\
               np.clip(temp[2], 0, im.shape[1]-1),\
               np.clip(temp[3], 0, im.shape[0]-1)]
        res[2] = res[2]-res[0]
        res[3] = res[3]-res[1]
        temp_result[i,:] = res
        res = [int(l) for l in res]
        #IOU with groundtruth

        pred_polygon = (res[0], res[1], res[0] + res[2], res[1],
                                    res[0] + res[2], res[1] + res[3], res[0],
                                    res[1] + res[3])
        gt = gtbbox
        if(len(gt) == 4):
            gt_polygon = (gt[0], gt[1], 
                          gt[0] + gt[2], gt[1],
                          gt[0] + gt[2], gt[1] + gt[3], 
                          gt[0], gt[1] + gt[3])
        else:
            gt_polygon = (gt[0], gt[1], gt[2], gt[3], gt[4], gt[5], gt[6], gt[7])
        iou_result[i] = vot_overlap(gt_polygon, pred_polygon,(im.shape[1], im.shape[0]))


        detection_score[i] = score[top_score[i]]
        score_result[i] = np.exp(-(r_c[top_score[i]] * s_c[top_score[i]] -1.) * p.penalty_k)

    score_result = score_result * score[top_score]
    score_result = score_result * (1 - p.window_influence) + window[top_score] * p.window_influence

    top_ids = score_result.argsort()[-20:][::-1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    best = []
    for j in range(19,-1,-1):
        #cv2.putText(im,"%.2f" % iou_result[top_ids[j]],(20,20+j*10), font, 0.4,(0,0,0), 1,cv2.LINE_AA)
        #cv2.putText(im,"%.2f" % score_result[top_ids[j]],(50,20+j*10), font, 0.4,(0,0,0), 1,cv2.LINE_AA)
        #cv2.putText(im,"%.2f" % detection_score[top_ids[j]],(220,20+j*10), font, 0.4,(0,0,0), 1,cv2.LINE_AA)
        res = temp_result[top_ids[j]]
        #if(detection_score[top_ids[j]] > 0.75):
        #    cv2.rectangle(im,  (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (255,255, 0), 3)
        #else:
        #    cv2.rectangle(im,  (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (255,0, 0), 3)

    target_pos = np.array([temp_result[top_ids[0],0] + (temp_result[top_ids[0],2]/2), temp_result[top_ids[0],1] + (temp_result[top_ids[0],3]/2)])
    target_sz = np.array([temp_result[top_ids[0],2],temp_result[top_ids[0],3]])
    alternative = []
    return target_pos, target_sz, score_result[top_ids[0]], iou_result, detection_score, temp_result, alternative

def SiamRPN_init(im, target_pos, target_sz, net):
    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]

    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271

        p.score_size = (p.instance_size - p.exemplar_size) / p.total_stride + 1

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
    s_z = round(np.sqrt(wc_z * hc_z))
    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)

    z = Variable(z_crop.unsqueeze(0))
    net.temple(z.cuda())

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    #[[cx,cy],[w,h]]
    state['fg'] = []
    state['detection_box'] = [int(target_pos[0]-target_sz[0]/2),int(target_pos[1]-target_sz[1]/2),int(target_pos[0]+target_sz[0]/2),int(target_pos[1]+target_sz[1]/2)]
    return state
def SiamRPN_track(state, im, gtbbox):
#def SiamRPN_track_bbox(state, im, next_mask, conf_mask, foreground_mask, entropy_mask, gtbbox):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = p.exemplar_size / s_z
    d_search = (p.instance_size - p.exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))

    target_pos, target_sz, score, iou_result, detection_score, temp_result, alternative =tracker_eval_box(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p, im, gtbbox)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    state['fg'] = alternative
    return state, iou_result, detection_score, temp_result


