import torchvision
import torch
import numpy as np
from BoxHead import *
from utils import *
from pretrained_models import *

if __name__ == '__main__':

    # Put the path were you save the given pretrained model
    pretrained_path='models/checkpoint680.pth'
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    backbone, rpn = pretrained_models_680(pretrained_path)

    # we will need the ImageList from torchvision
    from torchvision.models.detection.image_list import ImageList

    # Put the path were the given hold_out_images.npz file is save and load the images
    hold_images_path='hold_out_images.npz'
    test_images=np.load(hold_images_path,allow_pickle=True)['input_images']


    # Load your model here. If you use different parameters for the initialization you can change the following code
    # accordingly
    boxHead=BoxHead()
    boxHead=boxHead.to(device)
    boxHead.eval()

    # Put the path were you have your save network
    train_model_path='model_4B_v2_epoch36.pth'
    checkpoint = torch.load(train_model_path)
    # reload models
    boxHead.load_state_dict(checkpoint)
    keep_topK=200

    cpu_boxes = []
    cpu_scores = []
    cpu_labels = []

    for i, numpy_image in enumerate(test_images, 0):
        images = torch.from_numpy(numpy_image).to(device)
        with torch.no_grad():
            # Take the features from the backbone
            backout = backbone(images)

            # The RPN implementation takes as first argument the following image list
            im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
            # Then we pass the image list and the backbone output through the rpn
            rpnout = rpn(im_lis, backout)

            #The final output is
            # A list of proposal tensors: list:len(bz){(keep_topK,4)}
            proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
            # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
            fpn_feat_list= list(backout.values())


            feature_vectors=boxHead.MultiScaleRoiAlign(fpn_feat_list,proposals)

            class_logits,box_pred=boxHead(feature_vectors)

            # Do whaterver post processing you find performs best
            boxes,scores,labels=boxHead.postprocess_detections(class_logits,box_pred,proposals,conf_thresh=0.5, keep_num_preNMS=50, keep_num_postNMS=3)

            for box, score, label in zip(boxes,scores,labels):
                if box is None:
                    cpu_boxes.append(None)
                    cpu_scores.append(None)
                    cpu_labels.append(None)
                else:
                    cpu_boxes.append(box)
                    cpu_scores.append(score)
                    cpu_labels.append(label)

    np.savez('predictions.npz', predictions={'boxes': cpu_boxes, 'scores': cpu_scores,'labels': cpu_labels})
