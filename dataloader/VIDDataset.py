import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import numpy as np
import sys
import cv2
import json

sys.path.append('/home/jianingq/research_tool/datasets/vot/')
sys.path.append('/home/jianingq/research_tool/datasets/')

from normal_loader import VOT
from data_format import bbox_format
sys.path.append('/home/jianingq/bgflow/DaSiamRPN/code/jenny')
from utils import *

class MaskDataset(Dataset):
    def __init__(self):
        np.random.seed(0)
        self.train_data_pth = []
        self.test_data_pth = []
        self.tg_size = [128, 128]
        snaps = json.load(open('/home/jianingq/bgflow/DaSiamRPN/code/jenny/dataloader/vid/snippet.json', 'r'))
        all_frames = []
        for s, snap in enumerate(snaps):
            if(s>=1100):
                """
                print(s)
                frames = snap['frame']
                n_frames = len(frames)
                #print('subset: {} video id: {:04d} / {:04d}'.format(snap['base_path'], s, len(snaps)))

                for f, frame in enumerate(frames):
                    if(f>1):
                        img_path = os.path.join(snap['base_path'], frame['img_path'])
                        out_path = img_path.split('.')[0]
                        entropy = (out_path + '_entropy.npy')
                        confidence =out_path + '_confidence.npy'
                        lab = (out_path + '_lab.npy')
                        scale = (out_path + '_scale.npy')
                        flow = (out_path + '_flow.npy')
                        try:
                            os.remove(flow)
                            os.remove(entropy)
                            os.remove(confidence)
                            os.remove(lab)
                            os.remove(scale)
                            print("Removed {}".format(out_path))
                        except:
                            print("cannot remove {}".format(out_path))
                """
                break
                
            frames = snap['frame']
            n_frames = len(frames)
            #print('subset: {} video id: {:04d} / {:04d}'.format(snap['base_path'], s, len(snaps)))

            for f, frame in enumerate(frames):
                img_path = os.path.join(snap['base_path'], frame['img_path'])
                out_path = img_path.split('.')[0]
                frame_num = frame['img_path'].split('.')[0]
                #if(frame_num == 170):
                #    print(img_path)
                object_type = frame['obj']['c']
                bbox_dir = os.path.join(snap['base_path'],object_type)
                #x0 y0 x1 y1
                current_bbox = frame['obj']['bbox']
                frame['base_path'] = snap['base_path']
                if(f>1):
                    #if(frame_num == 170):
                    #    print(img_path)
                    #    print('inside')
                    #    print(f)
                    all_frames.append(frame)

        idx = np.random.permutation(len(all_frames))
        nTest = len(all_frames) // 3
        train_idx = idx
        test_idx = idx
        #train_idx = idx[nTest:]
        #test_idx = idx[:nTest]
        self.train_data_pth = [all_frames[elem] for elem in train_idx]
        self.test_data_pth = [all_frames[elem] for elem in test_idx]
        #self.train_data_pth.sort()
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

    def get_vid_image(self, idx, test=False):
        pth = self.train_data_pth[idx] if not test else self.test_data_pth[idx]
        img_path, frame_num, object_type, bbox_dir = self._info_from_pth(pth)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.tg_size[1], self.tg_size[0]))
        # cv2.rectangle(img,  (int(gt[0]),int(gt[1])), (int(gt[2]),int(gt[3])), (255,255, 0), 3)
        # cv2.imwrite('results/gt_mask.png', gt_mask[:,:,np.newaxis] * img)
        # cv2.imwrite('results/img.png', img)
        return img
    
    def _construct_mask(self, pth, prev=False):
        img_path, frame_num, object_type, bbox_dir = self._info_from_pth(pth)
        out_path = img_path.split('.')[0]
        # if frame_num > 3:
        #   return
        prev_frame_num = int(frame_num)-1
        #print(int(frame_num))
        flow = np.load(os.path.join(out_path+'_flow.npy'))
        print(flow.shape)
        if prev_frame_num == 0:
            x0, y0, x1, y1 = pth['obj']['bbox']
            mask = np.zeros([flow.shape[0], flow.shape[1]])
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            mask[y0:y1, x0:x1] = 1
        else:
            if not prev:
                bboxes, scores, det_scores = self._load_bbox(bbox_dir, format(prev_frame_num, '06'))
                mask = np.zeros([flow.shape[0], flow.shape[1]])
                for bbox, score in zip(bboxes, det_scores):
                    x1, y1, w, h = bbox
                    x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                    if(w<= 0 or h <= 0):
                        print(img_path)
                        print('zero size box')
                        continue
                    gaussian_mask = self._gaussian_box(h, w)
                    #mask[y1:y1+h, x1:x1+w] = np.maximum(mask[y1:y1+h, x1:x1+w],score*gaussian_mask)
                    mask[y1:y1+h, x1:x1+w] = np.maximum(mask[y1:y1+h, x1:x1+w],gaussian_mask)
            else:
                mask = np.load(os.path.join(bbox_dir,frame_num+'_fgmask_nd.npy'))
        # cv2.imwrite('../results/tests/new_mask.png', mask * 255)
        #rgb_img = cv2.imread(img_path)
        #cv2.imwrite('../results/tests/new_withoutwarped_mask.png', (mask+0.1)[:,:,np.newaxis] * rgb_img)
        mask = flow_warp(mask, flow)
        #mask = cv2.resize(mask, (self.tg_size[1], self.tg_size[0]))
        #cv2.imwrite('../results/tests/new_warped_mask.png', (mask+0.1)[:,:,np.newaxis] * rgb_img)

        #if not prev:
        #   print(pth+'_fgmask.npy')
        #   np.save(pth+'_fgmask.npy', mask)
        if not prev:
            print(os.path.join(bbox_dir,frame_num+'_fgmask_nd.npy'))
            np.save(os.path.join(bbox_dir,frame_num+'_fgmask_nd.npy'), mask)
        else:
            print(os.path.join(bbox_dir,frame_num+'_prev_fgmask_nd.npy'))
            np.save(os.path.join(bbox_dir,frame_num+'_prev_fgmask_nd.npy'), mask)

    def _info_from_pth(self, pth):
        img_path = os.path.join(pth['base_path'], pth['img_path'])
        frame_num = pth['img_path'].split('.')[0]
        object_type = pth['obj']['c']
        bbox_dir = os.path.join(pth['base_path'],object_type)
        return img_path, frame_num, object_type, bbox_dir

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

    def _load_bbox(self, bbox_dir,frame_num):
        pth = os.path.join(bbox_dir, frame_num+'.txt')
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
        img_path, frame_num, object_type, bbox_dir = self._info_from_pth(pth)
        out_path = img_path.split('.')[0]
        if int(frame_num) == 0:
            return None
        # load feature masks

        entropy = np.load(out_path + '_entropy.npy')
        confidence =np.load(out_path + '_confidence.npy')
        lab = np.load(out_path + '_lab.npy')
        
        fgmask = np.load(os.path.join(bbox_dir,frame_num+'_fgmask_nd.npy'))
        prev_fgmask_pth = os.path.join(bbox_dir,frame_num+ '_prev_fgmask_nd.npy')
        #fgmask = np.load(os.path.join(bbox_dir,frame_num+'_reverse_fgmask.npy'))
        #prev_fgmask_pth = os.path.join(bbox_dir,frame_num+ '_prev_reverse_fgmask.npy') 

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
        img_path, frame_num, object_type, bbox_dir = self._info_from_pth(pth)
        if int(frame_num) == 0:
            return None
        
        bboxes, scores, det_scores = self._load_bbox(bbox_dir,frame_num)
        masks = []
        tg_w, tg_h = self.tg_size
        all_feats = []
        for j, bbox in enumerate(bboxes):
            x, y, bw, bh = bbox
            mask = np.zeros([h,w])
            x, y, xp, yp = np.array([x, y, x+bw, y+bh])
            x, y, xp, yp = [int(k) for k in [x, y, xp, yp]]
            mask[y:yp, x:xp] = 1#det_scores[j]
            new_mask = cv2.resize(mask[y1:y2, x1:x2], (self.tg_size[1], self.tg_size[0]))
            # if j < 10:
            #   print(new_mask.shape)
            #   cv2.imwrite('../results/tests/bbox_{}.png'.format(j), new_mask*255)
            new_mask = Variable(torch.from_numpy(new_mask.copy()).float().cuda().unsqueeze(0).unsqueeze(0))
            all_feats.append(torch.cat((new_mask, feats), 1))
        
        return all_feats, scores


        """

        #rgb_img = cv2.imread(img_path)#
        bboxes, scores, det_scores = self._load_bbox(bbox_dir,frame_num)
        #bboxes, scores, det_scores = self._load_bbox(video_name, frame_num)
        tg_w, tg_h = self.tg_size
        all_feats = []
        for j, bbox in enumerate(bboxes):
            x, y, bw, bh = bbox
            mask = np.zeros([h,w])
            x, y, xp, yp = np.array([x, y, x+bw, y+bh])
            x, y, xp, yp = [int(k) for k in [x, y, xp, yp]]
            mask[y:yp, x:xp] = det_scores[j]

            x, y, xp, yp = self.zoom_out_det(x, y, xp, yp, mask)
            #mask = mask[:,:,np.newaxis] * rgb_img#
            new_mask = cv2.resize(mask[y:yp, x:xp], (self.tg_size[1], self.tg_size[0]))
            feats = [new_mask]
            #if j < 10:#
            #    cv2.imwrite('results/{}.jpg'.format(j), new_mask)#
            for i, mask in enumerate(masks):
                #if(len(mask.shape) == 3):
                #    mask = mask * rgb_img#
                #else:
                #    mask = mask[:,:,np.newaxis] * rgb_img#
                crop_mask = mask[y:yp, x:xp].copy()
                rsz_mask = cv2.resize(crop_mask, (self.tg_size[1], self.tg_size[0]))
                #if j < 10: #
                #    cv2.imwrite('results/{}_{}.jpg'.format(j,i), rsz_mask)#
                feats.append(rsz_mask)
            all_feats.append(Variable(torch.from_numpy(np.array(feats)).unsqueeze(0).float().cuda()))
        
        return all_feats, scores
        
        """
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
    #mask_ds.process(1000)
    # [4951, 13853, 4793]:
    #for i in range(len(mask_ds)):
    #    mask_ds.process(i)
    for i in range(47920,len(mask_ds)):
        print(i)
        mask_ds.process(i, prev=True)

def test_load_ds():
    mask_ds = MaskDataset()
    # mask_ds.process(4850)
    # example = mask_ds[4850]
    print(len(mask_ds.test_data_pth))
    for i in [100]:
        example = mask_ds.get_test_item(i)
        print(len(example['feats']))
        print(example['pth'])
        print(len(example['gt']), example['gt'].shape)
        # print(len(example['masks']), example['masks'][0].shape)
        print(example['feats'][0].shape)
        print(len(mask_ds), mask_ds.test_len())

if __name__ == '__main__':
    snaps = json.load(open('/home/jianingq/bgflow/DaSiamRPN/code/jenny/dataloader/vid/snippet.json', 'r'))
    np.random.seed(0)
    process_masks()
    #test_load_ds()
