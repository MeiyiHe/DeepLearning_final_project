import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
#from helper import convert_map_to_lane_map, convert_map_to_road_map
from CP_helper_RCNN import *

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]


# The dataset class for unlabeled data.
class UnlabeledDataset_RCNN(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data 
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform

        assert first_dim in ['sample', 'image']
        self.first_dim = first_dim

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE
    
    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 

            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                images.append(self.transform(image))
            image_tensor = torch.stack(images)
            
            m = transforms.Compose([transforms.Resize((800,800)), transforms.ToTensor()])
            comb_img = sew_images(image_tensor) 
            img =m(comb_img) #should be [3, 800, 800]
            
            return img, None #in the same format as img, target

        elif self.first_dim == 'image':
#             scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
#             sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
#             image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

#             image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name) 
            
#             image = Image.open(image_path)

#             return self.transform(image), index % NUM_IMAGE_PER_SAMPLE
              return None, None #it should never be called with 'image'



# The dataset class for labeled data.
class LabeledDataset_RCNN(torch.utils.data.Dataset):    
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=True):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data 
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """
        
        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
    
    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images) #should be [6, 3, 256, 306]
        
        m = transforms.Resize((800,800))
        comb_img = sew_images(image_tensor) 
        img =m(comb_img) #should be [3, 800, 800]
        
        data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
        
        categories = data_entries.category_id.to_numpy()
        
        #ego_path = os.path.join(sample_path, 'ego.png')
        #ego_image = Image.open(ego_path)
        #ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        #road_image = convert_map_to_road_map(ego_image)
        
        target = {}
        #target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        #target['category'] = torch.as_tensor(categories)
        
        #generate FastRNN target
        boxes = get_boxes(corners)
        labels = convert_categories(categories)
        masks = gen_masks( corners , labels) 
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) #this may need to be rounded but leave for now
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels)), dtype=torch.int64)
        target["boxes"]  = boxes
        target["labels"] = labels 
        target["masks"] = masks
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transform is not None:
            img = self.transform(img) #self.transforms is ToTensor()
            #target = self.transform(target) # keep target as a dictionary
        return img, target
        
        
# this is not needed        
#         if self.extra_info:
#             actions = data_entries.action_id.to_numpy()
#             # You can change the binary_lane to False to get a lane with 
#             lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)
            
#             extra = {}
#             extra['action'] = torch.as_tensor(actions)
#             extra['ego_image'] = ego_image
#             extra['lane_image'] = lane_image

#             return image_tensor, target, road_image, extra
        
#         else:
#          return image_tensor, target, road_image


        

#for i, bb in enumerate(target[0]['bounding_box']):
 
#draw_box(ax, bb, color=color_list[target[0]['category'][i]])
        
        
def get_boxes(corners): #this is the corners of the annotaion file
    # the corners are in meter and time 10 will convert them in pixels
    # Add 400, since the center of the image is at pixel (400, 400)
    # The negative sign is because the y axis is reversed for matplotlib
    #ax.plot(point_squence.T[0] * 10 + 400, -point_squence.T[1] * 10 + 400, color=color)
    
    
    #['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']
    #translate this to boxes to the fastRNN format
    xvals = corners[:, :4] *10 +400
    yvals = -(corners[:, 4:]*10 +400) #not flipping the y vals
    
    boxes = []
    num_obj = corners.shape[0]
    #print(corners.shape, num_obj)
    for i in range (num_obj):
        xmin = np.min(xvals[i])
        xmax = np.max(xvals[i])
        ymin = np.min(yvals[i])
        ymax = np.max(yvals[i])
    
        boxes.append([xmin, ymin, xmax, ymax])

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    
    return boxes
def convert_categories(categories):
    #Old categories
     
    # 'other_vehicle': 0,
    # 'bicycle': 1,
    # 'car': 2,
    # 'pedestrian': 3,
    # 'truck': 4,
    # 'bus': 5,
    # 'motorcycle': 6,
    # 'emergency_vehicle': 7, 
    # 'animal': 8
    
    #New categories
     
    # 'car': 1,
    # 'pedestrian': 2,
    # 'all other': 3,
     
    map_dict = {0:3, 1:3, 2:1, 3:2, 4:3, 5:3, 6:3, 7:3, 8:3}
    labels = []
    for c in categories:
        labels.append(map_dict[c])
    return torch.tensor(labels)

def gen_masks(corners , labels, img_w = 800, img_h = 800):
    '''
    essentially fill in the boxes in road_image with the class labels
    however all background is 0, hence no road is shown
    corners format: ['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']
    '''
    #print('corners shape {}'.format(corners.shape))
    corners = corners*10 +400 #convert into the road image format of 800, 800 with center being 400, 400
    xvals = np.round(corners[:, :4], 0).astype(int)
    yvals = -np.round(corners[:, 4:], 0).astype(int)
    num_obj = len(labels)
    #print('num_obj {}'.format(num_obj))
    masks = torch.zeros((num_obj, img_w, img_h))
    
    for i in range(num_obj):
        colmin = np.min(xvals[i])
        colmax = np.max(xvals[i])
        
        rowmin = np.min(yvals[i])
        rowmax = np.max(yvals[i])
        #print("mask shape {}".format(masks.shape))
        #print("i {}, xmin {}, xmax {}, ymin {}, ymax {} label {}".format(i, xmin, xmax, ymin, ymax, labels[i]))
        masks[i, rowmin:rowmax, colmin:colmax] = labels[i]
       
    return masks