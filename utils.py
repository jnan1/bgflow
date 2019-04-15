import numpy as np
# from skimage import color
import cv2

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def flow_warp(image, flow):
    pos = np.zeros(flow.shape, dtype=np.float32)
    pos[:, :, 0] = np.tile(np.arange(flow.shape[1]),
                           (flow.shape[0], 1)) + flow[:, :, 0]
    pos[:, :, 1] = np.tile(
        np.arange(flow.shape[0]).reshape(-1, 1),
        (1, flow.shape[1])) + flow[:, :, 1]
    return cv2.remap(image, pos, None, cv2.INTER_LINEAR)

# def calculate_iou(bbox1, bbox2):

#     # determine the coordinates of the intersection rectangle
#     x_left = max(bbox1[0], bbox2[0])
#     y_top = max(bbox1[1], bbox2[1])
#     x_right = min(bbox1[2], bbox2[2])
#     y_bottom = min(bbox1[3], bbox2[3])

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0

#     intersection_area = (x_right - x_left) * (y_bottom - y_top)

#     bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
#     bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

#     iou = intersection_area / \
#         float(bbox1_area + bbox2_area - intersection_area)
#     return iou

# def color_confidence(im1, im2): 
#     #switch from BGR to RGB
#     im1RGB = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
#     im2RGB = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
#     im1LAB =  color.rgb2lab(im1RGB)
#     im2LAB =  color.rgb2lab(im2RGB)
#     diff = color.deltaE_ciede2000(im1LAB,im2LAB)
#     dist = (im1RGB[:,:,0]-im2RGB[:,:,0])**2+(im1RGB[:,:,1]-im2RGB[:,:,1])**2 + (im1RGB[:,:,2]-im2RGB[:,:,2])**2
#     rgb_diff = dist
#     return diff, rgb_diff

# def nrmse(im1, im2):
#     im1 = np.array(im1)
#     im2 = np.array(im2)
#     a, b, c = im1.shape
#     rmse = np.sqrt(np.sum(np.sum((im2 - im1) ** 2))/ float(a * b))
#     max_val = max(np.max(im1), np.max(im2))
#     min_val = min(np.min(im1), np.min(im2))
#     return 1 - (rmse / (max_val - min_val))

# def background_flow(next_image, current_bbox, backward_flow):
#     mask = np.zeros(next_image.shape) + 0.3
#     mask[current_bbox[1]:current_bbox[3], current_bbox[0]:current_bbox[2], :] = 1
#     next_mask = flow_warp_backward(mask, backward_flow)

#     return next_mask

# #current_bbox: [x0, y0, x1, y2]
# def forward_warp(frame2, current_bbox, flow):
#     current_bbox[0] = np.clip(current_bbox[0], 0, frame2.shape[1]-1)
#     current_bbox[1] = np.clip(current_bbox[1], 0, frame2.shape[0]-1)
#     current_bbox[2] = np.clip(current_bbox[2], 0, frame2.shape[1]-1)
#     current_bbox[3] = np.clip(current_bbox[3], 0, frame2.shape[0]-1)
#     mask = np.zeros(frame2.shape).astype(int)
#     for j in range(current_bbox[1],current_bbox[3]-1):
#         for i in range(current_bbox[0],current_bbox[2]-1):
#             dx = flow[j,i,0]
#             dy = flow[j,i,1]
#             x = np.floor(i + dx).astype(int)
#             y = np.floor(j + dy).astype(int)
#             x = np.clip(x, 0, frame2.shape[1]-1)
#             y = np.clip(y, 0, frame2.shape[0]-1)
#             mask[y,x] = 1
#     kernel = np.ones((2,2),np.uint8)
#     mask_erosion = cv2.erode((mask).astype(np.uint8),kernel)
#     return mask_erosion

# def calculate_iou(bbox1, bbox2):

#     # determine the coordinates of the intersection rectangle
#     x_left = max(bbox1[0], bbox2[0])
#     y_top = max(bbox1[1], bbox2[1])
#     x_right = min(bbox1[2], bbox2[2])
#     y_bottom = min(bbox1[3], bbox2[3])

#     if x_right < x_left or y_bottom < y_top:
#         return 0.0

#     intersection_area = (x_right - x_left) * (y_bottom - y_top)

#     bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
#     bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

#     iou = intersection_area / \
#         float(bbox1_area + bbox2_area - intersection_area)
#     return iou

# #confidence_dir = os.path.join('/home/jianingq/backward_flow_confidence_vot/',video_name)
# #flow = np.load(os.path.join(confidence_dir,format(i+1, '08')+'_flow.npy'))
# def LAB_mask(im1,im2,flow):
#     warped_im1 = flow_warp_backward(im1, flow)
#     diff, rgb_diff = color_confidence(warped_im1, im2)
#     diff = diff/(np.max(diff)-np.min(diff))
#     diff_mask = np.zeros((np.shape(diff)[0],np.shape(diff)[1],3),dtype='float')
#     diff_mask[:,:,0] = diff
#     diff_mask[:,:,1] = diff
#     diff_mask[:,:,2] = diff
#     #np.save(os.path.join(data_dir,format(i+1, '08')+"_lab.npy"), diff_mask)
#     return diff_mask

# def confidence_mask()
#     entropy_data = np.load(os.path.join(confidence_dir,format(i+1, '08')+'_entropy.npy'))
#     entropy = (entropy_data - np.min(entropy_data))/(np.max(entropy_data)-np.min(entropy_data))
#     confidence = 1-entropy
#     entropy_mask = np.zeros((np.shape(entropy)[0],np.shape(entropy)[1],3),dtype='float')
#     entropy_mask[:,:,0] = entropy[:,:,0]
#     entropy_mask[:,:,1] = entropy[:,:,0]
#     entropy_mask[:,:,2] = entropy[:,:,0]
#     #np.save(os.path.join(data_dir,format(i+1, '08')+"_entropy.npy"), entropy_mask)

#     confidence_mask = np.zeros((np.shape(entropy)[0],np.shape(entropy)[1],3),dtype='float')
#     confidence_mask[:,:,0] = confidence[:,:,0]
#     confidence_mask[:,:,1] = confidence[:,:,0]
#     confidence_mask[:,:,2] = confidence[:,:,0]
#     #np.save(os.path.join(data_dir,format(i+1, '08')+"_confidence.npy"), confidence_mask)
#     return entropy_mask, confidence_mask