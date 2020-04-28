

## this file will be used to train road_map model and save .pt file

import os
import random

from collections import OrderedDict
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [5, 5]
matplotlib.rcParams['figure.dpi'] = 200

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torchvision.models as models


from data_helper import UnlabeledDataset, LabeledDataset
from helper import collate_fn, draw_box
from Unet import *


def my_collate_fn(batch):

	imgs = []
	road_imgs = []
	for x in batch:
		# first, stack the 6 images into 1    
		front = torch.cat( (torch.tensor( x[0][0] ), torch.tensor( x[0][1] ), torch.tensor( x[0][2] )), 2 )
		back = torch.cat( (torch.tensor( x[0][3] ), torch.tensor( x[0][4] ), torch.tensor( x[0][5] )), 2 )
		curr_image = torch.cat( (front, back), 1).transpose(2,1).flip(2)

		trans = transforms.Compose([
			transforms.ToPILImage(), 
			transforms.Resize((800,800)),
			transforms.ToTensor()])
		comb = trans(curr_image)
		imgs.append(comb.squeeze(0))
		
		# append current road map 
		road_imgs.append(x[2])

	return torch.stack(imgs), torch.stack(road_imgs)



if __name__ == '__main__':
	image_folder = '../data'
	annotation_csv = '../data/annotation.csv'
	
	random.seed(0)
	np.random.seed(0)
	torch.manual_seed(0);
	
	# You shouldn't change the unlabeled_scene_index
	# The first 106 scenes are unlabeled
	#unlabeled_scene_index = np.arange(106)

	# The scenes from 106 - 133 are labeled
	# You should devide the labeled_scene_index into two subsets (training and validation)
	labeled_scene_index = np.arange(106, 134)

	train_index = np.arange(106,128)
	val_index = np.arange(128,134)

	transform = transforms.Compose([
		transforms.Grayscale(num_output_channels=1),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.5],
			std=[0.5])
		])


	labeled_trainset = LabeledDataset(
		image_folder=image_folder,
		annotation_file=annotation_csv,
		scene_index=train_index,
		transform=transform,
		extra_info=True
		)

	labeled_valset = LabeledDataset(
		image_folder=image_folder,
		annotation_file=annotation_csv,
		scene_index=val_index,
		transform=transform,
		extra_info=True
		)

	trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=2, shuffle=False, num_workers=2, collate_fn=my_collate_fn)
	valloader = torch.utils.data.DataLoader(labeled_valset, batch_size=2, shuffle=False, num_workers=2, collate_fn=my_collate_fn)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#device = torch.device('cpu')
	model = UNet(in_channel=1,out_channel=1).to(device)
	for param in model.parameters():
		param.requires_grad = True

	criterion = torch.nn.BCELoss()
	param_list = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.Adam(param_list, lr=1e-4)
	best_val_loss = 100

	epochs = 10
	for epoch in range(epochs):

		#### train logic ####
		model.train()
		train_losses = []

		for i, (sample, road_img) in enumerate(trainloader):
			
			optimizer.zero_grad()
			
			sample = sample.to(device)
			road_img = road_img.to(device)

			pred_map = model(sample.unsqueeze(1))
			pred_map = (pred_map.squeeze(1)>0.5).float()
			pred_map.requires_grad = True

			loss = criterion(pred_map, road_img.float())
			train_losses.append(loss.item())

			loss.backward()
			optimizer.step()
			
			if i % 10 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, i * len(sample), len(trainloader.dataset),
					10. * i / len(trainloader), loss.item()))

		print("\n Average Train Epoch Loss for epoch {} is {} ", epoch+1, np.mean(train_losses))


		#### validation logic ####
		model.eval()
		val_losses = []

		for i, (sample, road_img) in enumerate(valloader):
			sample = sample.to(device)
			road_img = road_img.to(device)

			with torch.no_grad():
				pred_map = model(sample.unsqueeze(1))
				pred_map = (pred_map.squeeze(1)>0.5).float()
				loss = criterion(pred_map, road_img.float())
				val_losses.append(loss.item())

			if i % 10 == 0:
				print('Validation Epoch: {} [{}/{} ({:.0f}%)]\tAverage Loss So Far: {:.6f}'.format(
					epoch, i * len(sample), len(valloader.dataset),
					5. * i / len(valloader), np.mean(val_losses)))

			print("Average Validation Epoch Loss: ", np.mean(val_losses))

			if np.mean(val_losses) < best_val_loss:
				best_val_loss = np.mean(val_losses)
				torch.save(model.state_dict(), 'best_val_loss_road_map_labeleddata.pt')

