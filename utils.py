import numpy as np
import torch
from functools import partial
def MultiApply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
  
    return tuple(map(list, zip(*map_results)))


# This function decodes the output of the box head that are given in the [t_x,t_y,t_w,t_h] format
# into box coordinates where it return the upper left and lower right corner of the bbox
# Input:
#       regressed_boxes_t: (total_proposals,4) ([t_x,t_y,t_w,t_h] format)
#       flatten_proposals: (total_proposals,4) ([x1,y1,x2,y2] format)
# Output:
#       box: (total_proposals,4) ([x1,y1,x2,y2] format)
def output_decoding(boxes_pred,proposals, device='cpu'):
       #######################################
    # TODO decode the output
    #######################################
    boxes_pred = torch.tensor(boxes_pred).to(device)
    proposals = torch.tensor(proposals).to(device)
    box = torch.zeros(boxes_pred.shape)
    
    w = torch.exp(boxes_pred[:,2])*torch.abs(torch.tensor(proposals[:,0] - proposals[:,2]))
    h = torch.exp(boxes_pred[:,3])*torch.abs(torch.tensor(proposals[:,1] - proposals[:,3]))
    x = boxes_pred[:,0]*(torch.abs(torch.tensor(proposals[:,0] - proposals[:,2]))) + (proposals[:,0] + proposals[:,2])/2
    y = boxes_pred[:,1]*(torch.abs(torch.tensor(proposals[:,1] - proposals[:,3]))) + (proposals[:,1] + proposals[:,3])/2
    box[:,0] = torch.clip(x - w/2,min = 0)
    box[:,1] = torch.clip(y - h/2,min = 0)
    box[:,2] = torch.clip(x + w/2,max = 1088)
    box[:,3] = torch.clip(y + h/2,max = 800)
    
    return box


# This function computes the IOU between two set of boxes
def IOU(boxA, boxB):
    ##################################
    # computes the IOU between the boxA, boxB boxes
    ##################################
    top_right = torch.min(boxA[[2, 3]], boxB[:,[2, 3]])
    bot_left = torch.max(boxA[[0, 1]], boxB[:,[0, 1]])
    intersection = ((top_right - bot_left).clamp(min=0)).prod(dim=1)
    union = (boxA[2] - boxA[0])*(boxA[3] - boxA[1]) + (boxB[:,2] - boxB[:,0])*(boxB[:,3] - boxB[:,1]) - intersection + 1e-6
    iou = intersection/union
    return iou


def IOU_vectorized(bboxes, anchors):
    ##################################
    # computes the IOU between the bounding boxes, anchors
    ##################################
    boxA = bboxes.reshape((1, bboxes.shape[0], bboxes.shape[1]))
    boxB = anchors.reshape((anchors.shape[0], 1, anchors.shape[1]))

    x1 = boxA[:,:,0]
    y1 = boxA[:,:,1]
    x2 = boxA[:,:,2]
    y2 = boxA[:,:,3]
    
    x3 = boxB[:,:,0]
    y3 = boxB[:,:,1]
    x4 = boxB[:,:,2]
    y4 = boxB[:,:,3]

    x1_int = np.maximum(x1, x3)
    y1_int = np.maximum(y1, y3)
    x2_int = np.minimum(x2, x4)
    y2_int = np.minimum(y2, y4)
    int_area = np.clip(x2_int - x1_int,0,None) * np.clip(y2_int - y1_int,0,None)
    u_area = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - int_area
    with np.errstate(divide='ignore'):
        iou = int_area / u_area
        iou[int_area == 0] = 0
    iou = iou.reshape((anchors.shape[0], bboxes.shape[0]))
    return torch.from_numpy(iou)





