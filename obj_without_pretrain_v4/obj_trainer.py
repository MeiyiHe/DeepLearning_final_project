import os
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from anchors import get_bbox_gt, batched_coor_threat_updated
from helper import compute_ats_bounding_boxes


matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")


# define base trainer 
class Trainer:
    def __init__(self, model, optimizer, scheduler, trainloader, valloader, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.scheduler = scheduler
        self.device = device

    def train(self):
        raise NotImplementedError
    
    def validate(self):
        raise NotImplementedError


# construct anchor boxes
def get_anchor_boxes(scaleX=[100, 70, 50, 20], scaleY=[25, 20, 15, 5]):
    widths = torch.tensor(scaleX)
    heights = torch.tensor(scaleY)
    ref_boxes = []
    for x in range(800):
        for y in range(800):
            x_r = widths + x
            y_r = heights + y
            x_l = torch.tensor([x, x, x, x])
            y_l = torch.tensor([y, y, y, y])
            x_r = x_r.unsqueeze(0)
            y_r = y_r.unsqueeze(0)
            x_l = x_l.unsqueeze(0)
            y_l = y_l.unsqueeze(0)
            ref_box = torch.cat((x_l, y_l, x_r, y_r))
            ref_box = ref_box.permute((1,0))
            ref_boxes.append(ref_box)

    anchor_boxes = torch.stack(ref_boxes).view(-1,4).type(torch.double)
    
    return anchor_boxes


# object detection trainer
class ObjDetTrainer(Trainer):
    def __init__(self, model, optimizer, scheduler, trainloader, valloader, device="cpu", scaleX=[100, 70, 50, 20], scaleY=[25, 20, 15, 5]):
        super().__init__(model, optimizer, scheduler, trainloader, valloader, device)
        self.scaleX = scaleX
        self.scaleY = scaleY
        self.map_sz = 800
        self.img_h = 256
        self.img_w = 306
        self.batch_sz = 2
        self.anchor_boxes = self.get_anchor_boxes()
        #self.model.load_state_dict(torch.load('/content/drive/My Drive/self_dl/pre_train_subsample/epoch_4', map_location=self.device))
        self.model = self.model.to(self.device)
        self.best_val_loss = 100
        self.best_TS = -1

    def get_anchor_boxes(self):
        widths = torch.tensor(self.scaleX)
        heights = torch.tensor(self.scaleY)
        ref_boxes = []
        for x in range(self.map_sz):
            for y in range(self.map_sz):
                x_r = widths + x
                y_r = heights + y
                x_l = torch.tensor([x, x, x, x])
                y_l = torch.tensor([y, y, y, y])
                x_r = x_r.unsqueeze(0)
                y_r = y_r.unsqueeze(0)
                x_l = x_l.unsqueeze(0)
                y_l = y_l.unsqueeze(0)
                ref_box = torch.cat((x_l, y_l, x_r, y_r))
                ref_box = ref_box.permute((1,0))
                ref_boxes.append(ref_box)
    
        anchor_boxes = torch.stack(ref_boxes).view(-1,4).type(torch.double).to(self.device)
        
        return anchor_boxes


    def get_targets(self, target, sample):
        batched_preds = []
        batched_offsets = []
        for t, s in zip(target, sample):
            bboxes = t['bounding_box'].to(self.device)
            gt_classes, gt_offsets = get_bbox_gt(bboxes, t['category'].to(self.device), self.anchor_boxes.to(self.device), self.map_sz, self.device)
            batched_preds.append(gt_classes)
            batched_offsets.append(gt_offsets)

        class_targets = torch.stack(batched_preds)
        box_targets = torch.stack(batched_offsets)

        return class_targets, box_targets


    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()


    def bbox_loss(self, box_targets, class_targets, out_bbox):
        inds = (class_targets != 0)
        box_targets = box_targets[inds]
        out_bbox = out_bbox[inds]
	#criterion = torch.nn.MSELoss()
        criterion = torch.nn.MSELoss()
        loss_bbox = criterion(out_bbox, box_targets)

        return loss_bbox


    def train(self, epoch, save=True):
        print('Started train')
        for ep in range(epoch):
            self.model.train()
            print('Started epoch', ep)
            train_losses = []
            for i, (sample, target, road_image, extra) in enumerate(self.trainloader):
                samples = torch.stack(sample).to(self.device).double()
                #samples = samples.view(self.batch_sz, -1, 256, 306)
                #print('samples shape {}'.format(samples.shape))
                class_target, box_target = self.get_targets(target, sample)

                out_pred, out_bbox = self.model(samples)
                out_bbox = out_bbox.view(self.batch_sz, -1, 4)

                loss = self.bbox_loss(box_target, class_target, out_bbox)
                train_losses.append(loss.item())
          
                if loss.item() != 0:
                  self.step(loss)

                if i % 20  == 0:
                  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(ep, i * len(samples), len(self.trainloader.dataset), 10. * i / len(self.trainloader), loss.item()))
                  #self.validate(ep) 
                    
                torch.cuda.empty_cache()
 
            print("\nAverage Train Epoch Loss: ", np.mean(train_losses))
            self.validate(ep)

        #if save:
            #self.validate(ep, True, False)

    
    def validate(self, epoch, save=True, visualize=False):
        self.model.eval()
        val_losses = []
        threat_scores = []
        #coor_list = []
        print('Started validation')
        with torch.no_grad():
            for i, (sample, target, road_image, extra) in enumerate(self.valloader):
                samples = torch.stack(sample).to(self.device).double()
                #samples = samples.view(self.batch_sz, -1, 256, 306)
                class_target, box_target = self.get_targets(target, sample)
                out_pred, out_bbox = self.model(samples)
                out_bbox = out_bbox.view(self.batch_sz, -1, 4)


                for t in range(self.batch_sz):
                    pred_coor = get_coordinate(out_bbox[t], self.anchor_boxes, target[t]['bounding_box'].numpy(), class_target[t], nms_threshold=0.1, plot=False)
                    threat_scores.append( compute_ats_bounding_boxes( pred_coor, target[t]['bounding_box'] ).item() )
                    #coor_list.append(pred_coor)
                    #threat_scores.append( compute_ats_bounding_boxes(torch.stack(coordinate_list), target[t]['bounding_box'] ).item() )
               
                loss = self.bbox_loss(box_target, class_target, out_bbox)
                val_losses.append(loss.item())

                torch.cuda.empty_cache()

                if i % 20 == 0:
                    print('Val Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss So Far: {:.6f}'.format(epoch, i * len(samples), len(self.valloader.dataset), 5. * i / len(self.valloader), np.mean(val_losses)))
                
        print("Average Validation Epoch Loss: ", np.mean(val_losses))
        print("Average Threat Score: ", np.mean(threat_scores))

        if np.mean(val_losses) < self.best_val_loss:
            self.best_val_loss = np.mean(val_losses)


        if save and np.mean(threat_scores) > self.best_TS:
            self.best_TS = np.mean(threat_scores)
            print('== Saving model at epoch {} with best AVG Threat Score {} =='.format(epoch, self.best_TS))
            print('== Current Validation Loss is {} =='.format(np.mean(val_losses)))
            torch.save(self.model.state_dict(), 'bbox_no_pretrain03.pt')

        # if visualize:
        #     Transform_coor(out_bbox, gt_offsets, class_target, nms_threshold=0.1, plot=True)



