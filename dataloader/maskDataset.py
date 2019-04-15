import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import numpy as np
import sys
import cv2

sys.path.append('/home/jianingq/research_tool/datasets/vot/')
sys.path.append('/home/jianingq/research_tool/datasets/')

from normal_loader import VOT
from data_format import bbox_format
sys.path.append('/home/jianingq/bgflow/DaSiamRPN/code/jenny')
from utils import *
class MaskDataset(Dataset):
    def __init__(self, 
        # data_dir='/scratch/jianingq/vot_info', 
        data_dir='/home/jianingq/vot_info', 
        bbox_dir='/home/jianingq/vot/bbox_more_info/', 
        vot_root = '/scratch/jianren/vot_data',
        # vot_root='/scratch/jianingq/vot_data/',
        # flow_dir='/scratch/jianingq/backward_flow_confidence_vot/'):
        flow_dir='/home/jianingq/backward_flow_confidence_vot/'):
        np.random.seed(0)
        self.data_dir = data_dir
        self.train_data_pth = []
        self.test_data_pth = []
        self.vot = VOT(vot_root)
        self.gts = {}
        self.tg_size = [128, 128]
        self.bbox_dir = bbox_dir
        self.flow_dir = flow_dir

        #videos = [os.path.join(self.data_dir, x) for x in os.listdir(self.data_dir)]
        #videos = ['ants3', 'frisbee']
        #videos = ['iceskater2', 'iceskater1','ants3', 'frisbee']


        videos = ['ants1','bag', 'ball1', 'ball2', 'basketball', 'birds1',
            'blanket', 'bmx', 'bolt1', 'bolt2', 'book', 'butterfly', 'car1',
            'conduction1', 'crabs1', 'crossing', 'dinosaur', 'drone_across',
            'drone_flip', 'drone1', 'fernando', 'fish1', 'fish2', 'fish3',
            'flamingo1', 'girl', 'glove', 'godfather', 'graduate',
            'gymnastics1', 'gymnastics2', 'gymnastics3', 'hand', 'handball1',
            'handball2', 'helicopter',  'leaves',
            'matrix', 'motocross1', 'motocross2', 'nature', 'pedestrian1',
            'rabbit', 'racing', 'road', 'shaking', 'sheep', 'singer2',
            'singer3', 'soccer1', 'soccer2', 'soldier', 'tiger', 'traffic',
            'wiper', 'zebrafish1']
        """
        videos = ['ball1', 'basketball', 'birds1','bmx', 'book', 'butterfly', 'fish2', 'glove', 'graduate',
            'gymnastics2', 'gymnastics3',  'handball1',
            'handball2',  'motocross1',  'pedestrian1',
             'road',  'sheep', 'singer2',
            'soccer2']
        """
        #videos = ['graduate']

        videos = [os.path.join(self.data_dir, x) for x in videos]

        videos = [x for x in videos if os.path.isdir(x)]
        video_names = [x.split('/')[-1] for x in videos]
        for video_name in video_names:
            self.gts[video_name] = self.vot.get_gts(video_name)
        idx = np.random.permutation(len(videos))
        nTest = len(videos) // 3
        # nTest = 0
        train_idx = idx[nTest:]
        #train_idx = idx
        test_idx = idx[:nTest]
        for ind, video_dir in enumerate(videos):
            video_name = video_dir.split('/')[-1]#
            video_length = self.vot.get_frame_length(video_name)#
            #frame_ids = [x.split('_')[0] for x in os.listdir(video_dir) if x.endswith('.npy')]
            frame_ids = [format(xx, '08') for xx in range(0,video_length)]
            frames = [os.path.join(video_dir, x) for x in set(frame_ids) if not int(x)==0]
            if ind in train_idx:
                self.train_data_pth += frames
            else:
                print(video_name)
                self.test_data_pth += frames
        self.train_data_pth.sort()
        #self.test_data_pth.sort()
        self.test_data_pth = self.train_data_pth
        
    def __len__(self):
        return len(self.train_data_pth)

    def __getitem__(self, idx):
        pth = self.train_data_pth[idx]
        # print(pth)
        # self._construct_mask(pth)
        return self._load_pth(pth)

    def process(self, idx, prev=False):
        pth = self.train_data_pth[idx]
        #pth = self.test_data_pth[idx]
        self._construct_mask(pth, prev)

    def test_len(self):
        return len(self.test_data_pth)

    def get_test_item(self, idx):
        pth = self.test_data_pth[idx]
        return self._load_pth(pth)

    def get_vot_image(self, idx, test=False):
        pth = self.train_data_pth[idx] if not test else self.test_data_pth[idx]
        video_name, frame_num = self._info_from_pth(pth)
        img = self.vot.get_frames(video_name)[frame_num]
        img = cv2.resize(img, (self.tg_size[1], self.tg_size[0]))
        # cv2.rectangle(img,  (int(gt[0]),int(gt[1])), (int(gt[2]),int(gt[3])), (255,255, 0), 3)
        # cv2.imwrite('results/gt_mask.png', gt_mask[:,:,np.newaxis] * img)
        # cv2.imwrite('results/img.png', img)
        return img
    
    def _construct_mask(self, pth, prev=False):
        video_name, frame_num = self._info_from_pth(pth)
        # if frame_num > 3:
        #   return
        prev_frame_num = frame_num-1
        flow = np.load(os.path.join(self.flow_dir, video_name, 
            format(frame_num, '08')+'_flow.npy'))
        print(flow.shape)
        if prev_frame_num == 0:
            bbox = self.gts[video_name][prev_frame_num]
            x0, y0, x1, y1 = bbox_format(bbox,'tlxy_wh_2_rect')
            mask = np.zeros([flow.shape[0], flow.shape[1]])
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            mask[y0:y1, x0:x1] = 1
        else:
            if not prev:
                bboxes, scores, det_scores = self._load_bbox(video_name, prev_frame_num)
                mask = np.zeros([flow.shape[0], flow.shape[1]])
                for bbox, score in zip(bboxes, det_scores):
                    x1, y1, w, h = bbox
                    x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                    gaussian_mask = self._gaussian_box(h, w)
                    mask[y1:y1+h, x1:x1+w] = np.maximum(mask[y1:y1+h, x1:x1+w],score*gaussian_mask)
            else:
                mask = np.load(os.path.join(self.data_dir, video_name, 
                    format(prev_frame_num, '08')+'_fgmask.npy'))
        # cv2.imwrite('../results/tests/new_mask.png', mask * 255)
        mask = flow_warp(mask, flow)
        # rgb_img = self.vot.get_frames(video_name)[frame_num]
        # mask = cv2.resize(mask, (self.tg_size[1], self.tg_size[0]))
        # cv2.imwrite('../results/tests/new_warped_mask.png', (mask+0.1)[:,:,np.newaxis] * rgb_img)
        #if not prev:
        #   print(pth+'_fgmask.npy')
        #   np.save(pth+'_fgmask.npy', mask)
        if not prev:
            print(pth+'_fgmask.npy')
            np.save(pth+'_fgmask.npy', mask)
        else:
            print(pth+'_prev_fgmask.npy')
            np.save(pth+'_prev_fgmask.npy', mask)

    def _info_from_pth(self, pth):
        video_name = pth.split('/')[-2]
        frame_num = int(pth.split('/')[-1])
        return video_name, frame_num

    def _load_pth(self, pth):
        # print(pth)
        # feats, masks, scores = self._load_masks(pth)
        feats, scores = self._load_masks(pth)

        sample = {
            'feats': feats,
            'gt': scores, 
            # 'masks': masks, 
            'pth': pth,
        }
        return sample

    def _load_bbox(self, video_name, frame_num):
        suffix = '{}.txt'.format(format(frame_num, '08'))
        pth = os.path.join(self.bbox_dir, video_name, suffix)
        f = open(pth, 'r')
        bboxes, det_scores, scores = [], [], []
        for line in f:
            nums = line.split(' ')
            bboxes.append([float(x) for x in nums[-5:-1]])
            scores.append(float(nums[-1]))
            det_scores.append(float(nums[0]))
            # scores.append(float(nums[-1]))
        # print(bboxes)
        return (bboxes, 
            Variable(torch.from_numpy(np.array(scores)).cuda().float()).unsqueeze(1), 
            det_scores)

    def _load_masks(self, pth):
        video_name, frame_num = self._info_from_pth(pth)
        if frame_num == 0:
            return None
        # load feature masks
        entropy = np.load(os.path.join(self.flow_dir, video_name, format(frame_num, '08') + '_entropy.npy'))
        confidence = 1 - entropy.copy()
        lab = np.load(pth + '_lab.npy')
        
        fgmask = np.load(pth + '_fgmask.npy')
        prev_fgmask_pth = pth + '_prev_fgmask.npy'
        #fgmask = np.load(pth + '_reverse_fgmask.npy')
        #prev_fgmask_pth = pth + '_prev_reverse_fgmask.npy'

        if os.path.exists(prev_fgmask_pth):
            prev_fgmask = np.load(prev_fgmask_pth)
        else:
            prev_fgmask = fgmask.copy()
        
        
        masks = [entropy, confidence, lab, fgmask, prev_fgmask]
        # fs = ['entropy', 'confidence', 'lab', 'prev_fgmask', 'fgmask']                         
        # for i in range(len(masks)):
        #   cv2.imwrite('../results/tests/{}.png'.format(fs[i]), masks[i] * 255)
        h, w = masks[0].shape[:2]

        #crop by fg mask
        try:
            x1, y1, x2, y2 = self._crop_out_roi(*masks[-2:])
        except:
            print('problem encountered at {}'.format(pth))
            return None
        feats = []
        for i, mask in enumerate(masks):
            crop_mask = mask[y1:y2, x1:x2].copy()
            new_mask = cv2.resize(crop_mask, (self.tg_size[1], self.tg_size[0]))
            feats.append(new_mask)

        feats = np.array(feats).copy()
        
        # load detections
        feats = Variable(torch.from_numpy(feats).float().cuda().unsqueeze(0))
        video_name, frame_num = self._info_from_pth(pth)
        if frame_num == 0:
            return None
        bboxes, scores, det_scores = self._load_bbox(video_name, frame_num)
        masks = []
        tg_w, tg_h = self.tg_size
        all_feats = []
        for j, bbox in enumerate(bboxes):
            x, y, bw, bh = bbox
            mask = np.zeros([h,w])
            x, y, xp, yp = np.array([x, y, x+bw, y+bh])
            x, y, xp, yp = [int(k) for k in [x, y, xp, yp]]
            mask[y:yp, x:xp] = det_scores[j]
            new_mask = cv2.resize(mask[y1:y2, x1:x2], (self.tg_size[1], self.tg_size[0]))
            # if j < 10:
            #   print(new_mask.shape)
            #   cv2.imwrite('../results/tests/bbox_{}.png'.format(j), new_mask*255)
            new_mask = Variable(torch.from_numpy(new_mask.copy()).float().cuda().unsqueeze(0).unsqueeze(0))
            all_feats.append(torch.cat((new_mask, feats), 1))
        
        return all_feats, scores
        
        '''
        # rgb_img = self.vot.get_frames(video_name)[frame_num]
        bboxes, scores, det_scores = self._load_bbox(video_name, frame_num)
        tg_w, tg_h = self.tg_size
        all_feats = []
        for j, bbox in enumerate(bboxes):
            x, y, bw, bh = bbox
            mask = np.zeros([h,w])
            x, y, xp, yp = np.array([x, y, x+bw, y+bh])
            x, y, xp, yp = [int(k) for k in [x, y, xp, yp]]
            mask[y:yp, x:xp] = det_scores[j]

            x, y, xp, yp = self.zoom_out_det(x, y, xp, yp, mask)
            # mask = mask[:,:,np.newaxis] * rgb_img
            new_mask = cv2.resize(mask[y:yp, x:xp], (self.tg_size[1], self.tg_size[0]))
            feats = [new_mask]
            # if j < 10:
            #   cv2.imwrite('results/{}.jpg'.format(j), new_mask)
            for i, mask in enumerate(masks):
                # mask = mask[:,:,np.newaxis] * rgb_img
                crop_mask = mask[y:yp, x:xp].copy()
                rsz_mask = cv2.resize(crop_mask, (self.tg_size[1], self.tg_size[0]))
                # if j < 10: 
                #   cv2.imwrite('results/{}_{}.jpg'.format(j,i), rsz_mask)
                feats.append(rsz_mask)
            all_feats.append(Variable(torch.from_numpy(np.array(feats)).unsqueeze(0).float().cuda()))
        
        return all_feats, scores
        '''

    def _gaussian_box(self, w, h):
        '''
        outputs w by h array
        '''
        std = max(w,h) // 2
        x, y = np.mgrid[(-w//2+1):(w//2+1), (-h//2+1):(h//2+1)]
        g = np.exp(-((x**2 + y**2)/(2*std**2)))
        return g / np.max(g.ravel())
        # return np.ones([w,h])

    def _crop_out_roi(self, mask1, mask2):
        bbox1 = bbox(mask1 > 0)
        bbox2 = bbox(mask2 > 0)
        y1,y2,x1,x2 = bbox1
        yp1,yp2,xp1,xp2 = bbox2
        xmin = min(x1, xp1)
        xmax = max(x2, xp2)
        ymin = min(y1, yp1)
        ymax = max(y2, yp2)
        xmid, ymid = (xmin + xmax) // 2,  (ymin + ymax) // 2
        w, h = (xmax-xmin) * 2, (ymax-ymin) * 2
        xmin, xmax = max(xmid - w // 2, 0), min(xmid + w//2, mask1.shape[1])
        ymin, ymax = max(ymid - h // 2, 0), min(ymid + h//2, mask1.shape[0])
        return int(xmin), int(ymin), int(xmax), int(ymax)

    def normalize_entropy(self, entropy, mean=0.0724658410441471, 
        std=2.491323154407462):
        return (entropy - mean) / std

    def zoom_out_det(self, x, y, xp, yp, mask, scale=2):
        xmin, ymin, xmax, ymax = x, y, xp, yp
        xmid, ymid = (xmin + xmax) // 2,  (ymin + ymax) // 2
        w, h = (xmax-xmin) * scale, (ymax-ymin) * scale
        xmin, xmax = max(xmid - w // 2, 0), min(xmid + w//2, mask.shape[1])
        ymin, ymax = max(ymid - h // 2, 0), min(ymid + h//2, mask.shape[0])

        x, y, xp, yp = [int(a) for a in [xmin, ymin, xmax, ymax]]
        return x, y, xp, yp

def process_masks():
    #compute-0-11
    mask_ds = MaskDataset()
    # mask_ds.process(4952)
    # [4951, 13853, 4793]:
    for i in range(len(mask_ds)):
        mask_ds.process(i)
    for i in range(len(mask_ds)):
        mask_ds.process(i, prev=True)

def test_load_ds():
    mask_ds = MaskDataset(data_dir='/home/jianingq/vot_info')
    # mask_ds.process(4850)
    # example = mask_ds[4850]
    print(len(mask_ds.test_data_pth))
    for i in [9883]:
        example = mask_ds.get_test_item(i)
        print(len(example['feats']))
        print(example['pth'])
        print(len(example['gt']), example['gt'].shape)
        # print(len(example['masks']), example['masks'][0].shape)
        print(example['feats'][0].shape)
        print(len(mask_ds), mask_ds.test_len())

if __name__ == '__main__':
    np.random.seed(0)
    process_masks()
    #test_load_ds()
