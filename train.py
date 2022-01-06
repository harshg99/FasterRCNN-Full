from BoxHead import *
from pretrained_models import pretrained_models_680
from dataset import BuildDataLoader, BuildDataset
import torch.optim as optim
import torch
import pdb

imgs_path = '../data/hw3_mycocodata_img_comp_zlib.h5'
masks_path = '../data/hw3_mycocodata_mask_comp_zlib.h5'
labels_path = '../data/hw3_mycocodata_labels_comp_zlib.npy'
bboxes_path = '../data/hw3_mycocodata_bboxes_comp_zlib.npy'
paths = [imgs_path, masks_path, labels_path, bboxes_path]
# load the data into data.Dataset
dataset = BuildDataset(paths)


# build the dataloader
# set 20% of the dataset as the training data
full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size
torch.random.manual_seed(1)    
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
batch_size = 1
train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
train_loader = iter(train_build_loader.loader())
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = iter(test_build_loader.loader())

epochs = 10
# Put the path were you save the given pretrained model
# pdb.set_trace()
pretrained_path='models/checkpoint680.pth'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
backbone, rpn = pretrained_models_680(pretrained_path, eval=True)

# Load your model here. If you use different parameters for the initialization you can change the following code
# accordingly
boxHead=BoxHead(device=device)
boxHead=boxHead.to(device)
optim = optim.SGD(boxHead.parameters(),lr = 0.01/batch_size,weight_decay=1.0e-4,momentum=0.90)

# # Put the path were you have your save network
# train_model_path='train_epoch39'
# checkpoint = torch.load(train_model_path)
# # reload models
# boxHead.load_state_dict(checkpoint['box_head_state_dict'])
# keep_topK=200

# cpu_boxes = []
# cpu_scores = []
# cpu_labels = []
# for e in range(epochs):
#     batch_loss = 0
#     batch_loss_c = 0
#     batch_loss_r = 0
#     print("Epoch {}".format(e))
#     for i, batch in enumerate(train_loader,0):
#         pdb.set_trace()
#         images = batch['images'][0,:,:,:]
#         gt_labels = batch['labels']
#         boxes = batch['bboxes']
#     # Take the features from the backbone
#         backout = backbone(images)

#         # The RPN implementation takes as first argument the following image list
#         im_lis = ImageList(images, [(800, 1088)]*images.shape[0])
#         # Then we pass the image list and the backbone output through the rpn
#         rpnout = rpn(im_lis, backout)

#         #The final output is
#         # A list of proposal tensors: list:len(bz){(keep_topK,4)}
#         proposals=[proposal[0:keep_topK,:] for proposal in rpnout[0]]
#         # A list of features produces by the backbone's FPN levels: list:len(FPN){(bz,256,H_feat,W_feat)}
#         fpn_feat_list= list(backout.values())


#         feature_vectors=boxHead.MultiScaleRoiAlign(fpn_feat_list,proposals)

#         class_logits,box_preds=boxHead(feature_vectors)
#         class_gt,class_gt_box = boxHead.create_ground_truth(proposals,gt_labels,boxes)
#         loss, loss_c, loss_r = boxHead.compute_loss(class_logits, box_preds, class_gt, class_gt_box,l=1,effective_batch=150)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#         batch_loss +=loss.cpu().item()
#         batch_loss_c += loss_c.cpu().item()
#         batch_loss_r += loss_r.cpu().item()
    
#     print("Total Loss, Loss Class, Loss R: {} {}  {}".format(batch_loss,batch_loss_c,batch_loss_r))
# torch.save(boxHead.state_dict(), './model_4B_v1.pth')