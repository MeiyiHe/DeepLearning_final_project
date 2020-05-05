#from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    '''
    Parameters: prediction (our output), inp_dim (input image dimension), anchors, num_classes, and an optional CUDA flag
    
    This function takes an detection feature map and turns it into a 2-D tensor, 
    where each row of the tensor corresponds to attributes of a bounding box.
    
    Example: for input image of 800*800, num_classes = 1, batchsize = 1,
             the prediction will be size [1, 39375, 6] 
            For each image in the batch, we have a 39375 * 6 table 
                                        row = bounding box, col = (4 bbox attributs, 1 obj scores, 1 class scores)
    '''
    #print('inp_dim', inp_dim)
    ### EDIT: only 1 class ###
    #num_classes = 1
    ##########################
    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
#     print('grid_size: ', grid_size)
#     print('num_anchors: ', num_anchors)
#     print('stride: ', stride)
#     print('batch_size: ', batch_size)
#     print('bbox attrs: ', bbox_attrs)
#     print('prediction shape: ', prediction.shape)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    
    # above transform feature map and turns into a 2D tensor
    ###########################################################################################

    # dimension of the anchors is corresponds to the height and width attributes of the net block
    # divide the anchors by stride of the detection feature map
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # transform the output
    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    # Resize the detection map to the size of the input image 800*800
    prediction[:,:,:4] *= stride
    
    return prediction


def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    '''
    Parameters: prediction, confidence (objectness score threshold), num_classes (1, in our case) and nms_conf (the NMS IoU threshold)    
    - prediction: [BatchZize, 39375, 6], 39375*6 = 39375 rows of (x,y,w,h,score,class) 
    
    Returns: a tensor of shape D x 8 
        (D: true detection in all of images, each represented by a row)
        (8 attributes, index of the image in the batch, 4 corner coordinates, objectness score, the score of class with max confidence, index of class)
    '''
    # for each of the bbox with score < than confidence score, we ignore by setting entire row to zero
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
   
    # TODO: transform to [N, 2, 4] format
    # transform (centerX, centerY, width, height) -> (top-left x, top-left y, bottom-right x, bottom-right y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    
    # loop thru the images in the batch
    batch_size = prediction.size(0)
    # indicate we haven't initialized the output
    # output is the tensor that will collect the True predictions across entire batch
    write = False
    
    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
        
        # confidence threshholding 
        # NMS: 
        # only concerned with the class score that have the max value, thus remove unused classes, kept only 1 class score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)

        # only kept nonzero indices
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        # handle situations where we don't have detections
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue        
  
        #Get the various classes detected in the image
        # since we only have 1 class, thus only could be 0
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        
        # for each class, perform NMS
        for cls in img_classes:
        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            #perform NMS
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try: # bbox_iou() takes (1) bbox row that indexed by the variable i, (2) a tensor of multiple rows of bounding boxes 
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    # ious: a tensor containing IoUs of the bounding box represented by the first input with each of the bbox remainings 
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      
            #Repeat the batch_id for as many detections of the class cls in the image
            
            seq = batch_ind, image_pred_class
            
            if not write:
                # don't initialize out until we have the 1st detection
                output = torch.cat(seq,1)
                write = True
            else:
                # once initialized, we concatenate subsequent detections
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    try:
        return output
    except:
        return 0
    
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names
