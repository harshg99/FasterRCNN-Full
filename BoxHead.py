import torch
import torch.nn.functional as F
from torch import nn
from utils import *
import numpy as np
import math
import torchvision
import pdb

class BoxHead(torch.nn.Module):
    def __init__(self,Classes=3,P=7):
        super(BoxHead,self).__init__()
        self.C=Classes
        self.P=P
        self.image_dim = [1088,800]
        # TODO initialize BoxHead
        self.intermediate = nn.Sequential(
            nn.Linear(in_features=256*self.P*self.P,out_features = 1024),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features = 1024),
            nn.ReLU())
        
        self.box_head = nn.Sequential( 
            nn.Linear(in_features=1024,out_features = self.C+1),
            nn.Softmax(dim=-1)
        )

        self.regressor_head = nn.Sequential( 
            nn.Linear(in_features=1024,out_features = 4*self.C)
        )


    #  This function assigns to each proposal either a ground truth box or the background class (we assume background class is 0)
    #  Input:
    #       proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #       gt_labels: list:len(bz) {(n_obj)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #  Output: (make sure the ordering of the proposals are consistent with MultiScaleRoiAlign)
    #       labels: (total_proposals,1) (the class that the proposal is assigned)
    #       regressor_target: (total_proposals,4) (target encoded in the [t_x,t_y,t_w,t_h] format)
    def create_ground_truth(self,proposals,gt_labels,bbox):
        num_props = proposals[0].shape[0]
        bz = len(proposals)
        labels = np.zeros(num_props * bz)
        regressor_target = np.zeros((num_props * bz, 4))

        for i, (props, gt_boxes, g_labels) in enumerate(zip(proposals, bbox, gt_labels)):
            # indices = torch.tensor([1,0,3,2]).to(props.device)
            # props = torch.index_select(props, 1, indices)
            props = props.clone().cpu().numpy()
            iou_scores = IOU_vectorized(props, gt_boxes)
            iou_scores = np.where(iou_scores > 0.5, iou_scores, 0)
            max_iou_idx = np.argmax(iou_scores, axis=0)
            l = g_labels[max_iou_idx]
            sums = np.sum(iou_scores, axis=0)
            l[sums == 0] = 0
            labels[(i * num_props):((i+1)*num_props)] = l
            
            bboxes = gt_boxes[max_iou_idx]
            for j, (a, b) in enumerate(zip(props, bboxes)):
                xp, yp, wp, hp = (a[0]+a[2])/2, (a[1]+a[3])/2, a[2] - a[0], a[3] - a[1]
                x, y, w, h = (b[0]+b[2])/2, (b[1]+b[3])/2, b[2] - b[0], b[3] - b[1]
                tx = (x - xp)/wp
                ty = (y - yp)/hp
                tw = np.log(w/wp)
                th = np.log(h/hp)
                regressor_target[(i*num_props) + j] = [tx, ty, tw, th]

        return labels,regressor_target



    # This function for each proposal finds the appropriate feature map to sample and using RoIAlign it samples
    # a (256,P,P) feature map. This feature map is then flattened into a (256*P*P) vector
    # Input:
    #      fpn_feat_list: list:len(FPN){(bz,256,H_feat,W_feat)}
    #      proposals: list:len(bz){(per_image_proposals,4)} ([x1,y1,x2,y2] format)
    #      P: scalar
    # Output:
    #      feature_vectors: (total_proposals, 256*P*P)  (make sure the ordering of the proposals are the same as the ground truth creation)
    def MultiScaleRoiAlign(self, fpn_feat_list,proposals,P=7):
        #####################################
        # Here you can use torchvision.ops.RoIAlign check the docs
        #####################################
        # assert np.abs(scale_x-scale_y)<=0.0
        self.P = P 
        proposals_fpn_level = [torch.floor(4 + torch.log2
        (1.0/224*torch.sqrt((proposals[i][:,2] - proposals[i][:,0])*(proposals[i][:,3] - proposals[i][:,1]))))
        for i in range(len(proposals))]
        feature_vectors = []
        for i in range(len(proposals)):
            # indices = torch.tensor([1,0,3,2]).to(proposals[i].device)
            # proposals[i] = torch.index_select(proposals[i], 1, indices)
            for j in range(proposals[i].shape[0]):
                level = int(proposals_fpn_level[i][j])
                #print(level)
                if level<2:
                    level = 2
                if level>5:
                    level = 5
                scale_x = self.image_dim[0]/fpn_feat_list[level-2].shape[2]
                scale_y = self.image_dim[1]/fpn_feat_list[level-2].shape[3]
                box_prop = torch.zeros(5).to(proposals[i].device)
                
                box_prop[1] = proposals[i][j][0]/scale_x
                box_prop[2] = proposals[i][j][1]/scale_y
                box_prop[3] = proposals[i][j][2]/scale_x
                box_prop[4] = proposals[i][j][3]/scale_y
                proposal_feature_vector = torchvision.ops.roi_align(fpn_feat_list[level-2][i].unsqueeze(0),box_prop.unsqueeze(0),output_size=(P,P),spatial_scale=1.0)
                feature_vectors.append(proposal_feature_vector.reshape(-1))
        feature_vectors  = torch.stack(feature_vectors,dim=0)
        return feature_vectors


    # This function does the post processing for the results of the Box Head for a batch of images
    # Use the proposals to distinguish the outputs from each image
    # Input:
    #       class_logits: (total_proposals,(C+1))
    #       box_regression: (total_proposal,4*C)           ([t_x,t_y,t_w,t_h] format)
    #       proposals: list:len(bz)(per_image_proposals,4) (the proposals are produced from RPN [x1,y1,x2,y2] format)
    #       conf_thresh: scalar
    #       keep_num_preNMS: scalar (number of boxes to keep pre NMS)
    #       keep_num_postNMS: scalar (number of boxes to keep post NMS)
    # Output:
    #       boxes: list:len(bz){(post_NMS_boxes_per_image,4)}  ([x1,y1,x2,y2] format)
    #       scores: list:len(bz){(post_NMS_boxes_per_image)}   ( the score for the top class for the regressed box)
    #       labels: list:len(bz){(post_NMS_boxes_per_image)}   (top class of each regressed box)
    def postprocess_detections(self, class_logits, box_regression, proposals, conf_thresh=0.5, keep_num_preNMS=500, keep_num_postNMS=50):
        preNMSperbatch = int(keep_num_preNMS/len(proposals))
        class_logits = class_logits.reshape((len(proposals),-1,class_logits.shape[1]))
        box_regression = box_regression.reshape((len(proposals),-1,box_regression.shape[1]))
        class_logits = torch.where(class_logits>0.5,class_logits,torch.tensor(0,dtype=torch.float32).to(class_logits.device)) 
        boxes_postnms = []
        scores_postnms = []
        labels_postnms = []
        for j in range(len(proposals)):    
            proposals_decoded = box_regression.clone()
            for k in range(3):
              proposals_decoded[j,:,4*k:4*(k+1)] = output_decoding(box_regression[j,:,4*k:4*(k+1)],proposals[j], device='cpu')
            class_logits[j,class_logits[j,:,0]!=0,0] = 0
            class_logit_filter = class_logits[j,torch.sum(class_logits[j,:,:],axis=1)!=0,:]
            boxes = proposals_decoded[j,torch.sum(class_logits[j,:,:],axis=1)!=0,:]
            cmax_sorted, indices = torch.sort(torch.max(class_logit_filter,dim=1)[0],dim=0,descending=True)
            c_sorted_top = class_logit_filter[indices[:preNMSperbatch]]
            boxes_top = boxes[indices[:preNMSperbatch]]
            boxes_top = boxes_top.reshape((boxes_top.shape[0],self.C,4))
            boxes_top_prenms = torch.zeros((boxes_top.shape[0],4))
            labels_top = torch.argmax(c_sorted_top,dim=1)
            for j, (label,box_pred) in enumerate(zip(labels_top, boxes_top)):
                boxes_top_prenms[j,:]  = box_pred[label-1]
            
            c_sorted_top_prenms,_ = torch.max(c_sorted_top,dim=1)
            #Do NMS on the boxes
            boxes_clas = np.array([])
            scores_clas = np.array([])
            labels_clas = np.array([])
            #pdb.set_trace()
            for k in range(self.C):
                if(torch.all(labels_top!=k+1)):
                    continue
                
                c_sorted_top_prenms[labels_top==k+1],boxes_top_prenms[labels_top==k+1] = self.NMS(c_sorted_top_prenms[labels_top==k+1],\
                boxes_top_prenms[labels_top==k+1],0.3)
                c_sorted_top_nms = c_sorted_top_prenms[torch.logical_and(c_sorted_top_prenms!=0,labels_top==k+1).cpu()]
                c_sorted_top_nms_ = c_sorted_top_nms[c_sorted_top_nms>=conf_thresh]
                scores_clas = np.concatenate((scores_clas,c_sorted_top_nms_[:keep_num_postNMS].detach().cpu().numpy()),axis = 0)
                boxes_top_nms = boxes_top_prenms[torch.logical_and(c_sorted_top_prenms!=0,labels_top==k+1).cpu()]
                boxes_top_nms = boxes_top_nms[c_sorted_top_nms>=conf_thresh]
                boxes_top_nms_ = boxes_top_nms[:keep_num_postNMS]
                if boxes_clas!=np.array([]):
                  boxes_clas = np.concatenate((boxes_clas,boxes_top_nms_.detach().cpu().numpy()),axis = 0)
                else:
                  boxes_clas = boxes_top_nms_.detach().cpu().numpy()
                
                labels_top_nms = labels_top[torch.logical_and(c_sorted_top_prenms!=0,labels_top==k+1).cpu()][c_sorted_top_nms>=conf_thresh]
                #print(labels_top)
                #print(labels_top_nms)
                #pdb.set_trace()
                labels_top_nms_ = labels_top_nms[:keep_num_postNMS]
                labels_clas = np.concatenate((labels_clas,labels_top_nms_.detach().cpu().numpy()),axis = 0)
            boxes_postnms.append(boxes_clas)
            scores_postnms.append(scores_clas)
            labels_postnms.append(labels_clas)

        return boxes_postnms, scores_postnms, labels_postnms


    
   # Input: 
    #       clas: (top_k_boxes) (scores of the top k boxes)
    #       prebox: (top_k_boxes,4) (coordinate of the top k boxes)
    # Output:
    #       nms_clas: (Post_NMS_boxes)
    #       nms_prebox: (Post_NMS_boxes,4)
    def NMS(self,clas,prebox,thresh):
        ##################################
        # TODO perform Nms
        ##################################
        nms_clas = []
        nms_prebox = []
        clas_copy = torch.clone(clas).detach().cpu().numpy()
        while True:
            if(np.count_nonzero(clas_copy)==0):
                break
            #print(clas_copy.shape)
            current_box = np.argwhere(clas_copy==clas_copy.max())
            box = torch.clone(prebox[current_box,:].squeeze())
            #print(box.shape)
            clas_copy[current_box] = 0
            iou = IOU(box,prebox)
            supp = torch.logical_and(iou>thresh,torch.abs(iou-1.0)>0.0001)
            # supresses all boxes that are above 0.5 iou
            clas_copy[supp] = 0
            clas[supp] = 0
        return clas,prebox

    # Compute the total loss of the classifier and the regressor
    # Input:
    #      class_logits: (total_proposals,(C+1)) (as outputed from forward, not passed from softmax so we can use CrossEntropyLoss)
    #      box_preds: (total_proposals,4*C)      (as outputed from forward)
    #      labels: (total_proposals,1)
    #      regression_targets: (total_proposals,4)
    #      l: scalar (weighting of the two losses)
    #      effective_batch: scalar
    # Outpus:
    #      loss: scalar
    #      loss_class: scalar
    #      loss_regr: scalar
    def compute_loss(self,class_logits, box_preds, labels, regression_targets,l=1,effective_batch=150):
        if labels[labels>0].shape[0] >= (3*effective_batch / 4):
            idx = np.random.choice(int(labels[labels>0].shape[0]), int(3*effective_batch/4),replace=False)
            class_logits_pos = class_logits[labels>0,:][idx]
            labels_pos = torch.tensor(labels[labels>0][idx],dtype=torch.int64).to(class_logits.device)
            regr_out_pos = box_preds[labels>0][idx]
            regr_out_pos= regr_out_pos.reshape((regr_out_pos.shape[0],self.C,4))
            targ_regr_pos = torch.tensor(regression_targets[labels>0,:][idx]).to(box_preds.device)
            idx = np.random.choice(int(labels[labels==0].shape[0]), int(effective_batch/4),replace=False)
            class_logits_neg = class_logits[labels==0,:][idx]
            labels_neg = torch.tensor(labels[labels==0][idx],dtype=torch.int64).to(class_logits.device)

        else:
            class_logits_pos = class_logits[labels>0,:]
            labels_pos = torch.tensor(labels[labels>0],dtype=torch.int64).to(class_logits.device)
            regr_out_pos = box_preds[labels>0]
            regr_out_pos= regr_out_pos.reshape((regr_out_pos.shape[0],self.C,4))
            targ_regr_pos = torch.tensor(regression_targets[labels>0,:]).to(box_preds.device)
            idx = np.random.choice(int(labels[labels==0].shape[0]), int(effective_batch - labels[labels>0].shape[0]),replace=False)
            class_logits_neg = class_logits[labels==0,:][idx]
            labels_neg = torch.tensor(labels[labels==0][idx],dtype=torch.int64).to(class_logits.device)
        
        loss_class = torch.sum(torch.log(class_logits_pos)*F.one_hot(labels_pos,self.C+1))+ torch.sum(torch.log(class_logits_neg)*F.one_hot(labels_neg,self.C+1))
        loss_class = - loss_class.sum()/effective_batch
        loss_func = torch.nn.SmoothL1Loss(reduction='sum')
        loss_regr_1 = torch.nan_to_num(torch.sum(loss_func(regr_out_pos[labels_pos==1,0,:],targ_regr_pos[labels_pos==1])),nan=0.0)
        loss_regr_2 = torch.nan_to_num(torch.sum(loss_func(regr_out_pos[labels_pos==2,1,:],targ_regr_pos[labels_pos==2])),nan=0.0)
        loss_regr_3 = torch.nan_to_num(torch.sum(loss_func(regr_out_pos[labels_pos==3,2,:],targ_regr_pos[labels_pos==3])),nan=0.0)
        loss_regr = (loss_regr_1+loss_regr_2+loss_regr_3)/effective_batch
        loss = loss_class + l * loss_regr
        return loss, loss_class, loss_regr



    # Forward the pooled feature vectors through the intermediate layer and the classifier, regressor of the box head
    # Input:
    #        feature_vectors: (total_proposals, 256*P*P)
    # Outputs:
    #        class_logits: (total_proposals,(C+1)) (we assume classes are C classes plus background, notice if you want to use
    #                                               CrossEntropyLoss you should not pass the output through softmax here)
    #        box_pred:     (total_proposals,4*C)
    def forward(self, feature_vectors):
        fv = self.intermediate(feature_vectors)
        class_logits = self.box_head(fv)
        box_pred = self.regressor_head(fv)
        return class_logits, box_pred

# if __name__ == '__main__':
    
