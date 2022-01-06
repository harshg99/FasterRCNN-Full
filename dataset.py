import torch
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from utils import *
import pdb

##Custom build of dataset
class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        #############################################
        # Initialize  Dataset
        #############################################
        self.masks = h5py.File(path[1])['data']
        self.img = h5py.File(path[0])['data']
        self.lab = np.array(np.load(path[2], allow_pickle=True))
        self.bbox = np.array(np.load(path[3],allow_pickle=True))

    # In this function for given index we rescale the image and the corresponding masks, boxes
    # and we return them as output
    # output:
        # transed_img
        # label
        # transed_mask
        # transed_bbox
        # index
    def __getitem__(self, idx): 
        ################################
        # return transformed images,labels,masks,boxes,index
        ################################
        mask_numb = 0
        mask_numb_end = 0      

        for i in range(idx):
            mask_numb+=len(self.lab[i])
            #print(len(labels[idx]))
        mask_numb_end = mask_numb + len(self.lab[idx])
        
        transed_img,transed_mask,transed_bbox = self.pre_process_batch(torch.tensor(self.img[idx]/255,dtype=torch.float32),
            torch.tensor(self.masks[mask_numb:mask_numb_end]/1.0,dtype=torch.int32),self.bbox[idx])

        #bbox[:,[1,3]] += 11 #no need to pad the bounding box
        #sample = img,self.lab[idx],masks,bbox
        label = self.lab[idx]
        index = idx

        assert transed_img.shape == (3,800,1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox, index

    # This function preprocess the given image, mask, box by rescaling them appropriately
    # output:
    #        img: (3,800,1088)
    #        mask: (n_box,800,1088)
    #        box: (n_box,4)
    def pre_process_batch(self, img, mask, bbox): 
        #######################################
        # TODO apply the correct transformation to the images,masks,boxes
        ######################################
        self.transform = transforms.Compose([
        transforms.Resize((800, 1066)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.Pad((11,0))])
        self.transform_mask = transforms.Compose([
        transforms.Resize((800, 1066)),
        transforms.Pad((11,0))])
        img = self.transform(img.clone().detach())
        mask = self.transform_mask(mask.clone().detach())
        bbox = bbox*2.667
        bbox[:,[1,3]] += 11
        assert img.squeeze(0).shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.shape[0]
        return img, mask, bbox
    
    def __len__(self):
        return len(self.img)




class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers): #The terms present here are to initialize, they are not to be returned
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers


    # output:
    #  dict{images: (bz, 3, 800, 1088)
    #       labels: list:len(bz)
    #       masks: list:len(bz){(n_obj, 800,1088)}
    #       bbox: list:len(bz){(n_obj, 4)}
    #       index: list:len(bz)
    def collect_fn(self, batch):
        img_list = []
        lab_list =[]
        mask_list = []
        bb_list =[]
        idx_list = []

        for images, labels, masks, bbox, index in batch:
            img_list.append(images)
            lab_list.append(labels)
            mask_list.append(masks)
            bb_list.append(bbox)
            idx_list.append(index)
        
        zipped = zip(img_list, lab_list, mask_list, bb_list, idx_list)
        out_batch =  dict()
        out_batch['images'] = torch.stack(img_list,dim=0)
        out_batch['labels'] = lab_list
        out_batch['masks'] = mask_list
        out_batch['bboxes'] = bb_list
        out_batch['indexes'] = idx_list
        return out_batch
    
    def loader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


#From here you set up the path of the dataset and then you send it over to the custom builder dataset
if __name__ == '__main__':
    imgs_path = './data/hw3_mycocodata_img_comp_zlib.h5'
    masks_path = './data/hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = './data/hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = './data/hw3_mycocodata_bboxes_comp_zlib.npy'
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

 