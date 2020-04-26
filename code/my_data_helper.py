import os
from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from helper import convert_map_to_lane_map, convert_map_to_road_map

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6

COMBINED_IMAGE_PER_SAMPLE = 1


image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]

def append_images(images, direction='horizontal', bg_color=(255,255,255), aligment='center'):
	"""
	Appends images in horizontal/vertical direction.
	Args:
		images: List of PIL images
		direction: direction of concatenation, 'horizontal' or 'vertical'
		bg_color: Background color (default: white)
		aligment: alignment mode if images need padding;
			'left', 'right', 'top', 'bottom', or 'center'
	Returns:
		Concatenated image as a new PIL image object.
	"""
	widths, heights = zip(*(i.size for i in images))
	if direction=='horizontal':
		new_width = sum(widths)
		new_height = max(heights)
	else:
		new_width = max(widths)
		new_height = sum(heights)

	new_im = Image.new('RGB', (new_width, new_height), color=bg_color)
	offset = 0

	for im in images:
		if direction=='horizontal':
			y = 0
			if aligment == 'center':
				y = int((new_height - im.size[1])/2)
			elif aligment == 'bottom':
				y = new_height - im.size[1]

			new_im.paste(im, (offset, y))
			offset += im.size[0]
		else:
			x = 0
			if aligment == 'center':
				x = int((new_width - im.size[0])/2)
			elif aligment == 'right':
				x = new_width - im.size[0]
			new_im.paste(im, (x, offset))
			offset += im.size[1]
	return new_im



# The dataset class for unlabeled data.
class UnlabeledDataset(torch.utils.data.Dataset):
	def __init__(self, image_folder, scene_index, first_dim, transform):
		"""
		Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data 
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, COMBINED_IMAGE_PER_SAMPLE=1, 3, H, W] 
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

			front = [ Image.open(os.path.join(sample_path, image_names[0])), Image.open(os.path.join(sample_path, image_names[1])), Image.open(os.path.join(sample_path, image_names[2])) ]
			back = [ Image.open(os.path.join(sample_path, image_names[3])), Image.open(os.path.join(sample_path, image_names[4])), Image.open(os.path.join(sample_path, image_names[5])) ]

			combo_1 = append_images(front, direction='horizontal')
			combo_2 = append_images(back, direction='horizontal')
			combined = append_images([combo_1, combo_2], direction='vertical')

			image_tensor = self.transform(combined)
			#return image_tensor
			return torch.stack([image_tensor])

		elif self.first_dim == 'image':
			scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
			sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
			image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

			image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name)
			image = Image.open(image_path)
			return self.transform(image), index % NUM_IMAGE_PER_SAMPLE

# The dataset class for labeled data.
class LabeledDataset(torch.utils.data.Dataset):
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

		front = [ Image.open(os.path.join(sample_path, image_names[0])), Image.open(os.path.join(sample_path, image_names[1])), Image.open(os.path.join(sample_path, image_names[2])) ]
		back = [ Image.open(os.path.join(sample_path, image_names[3])), Image.open(os.path.join(sample_path, image_names[4])), Image.open(os.path.join(sample_path, image_names[5])) ]

		combo_1 = append_images(front, direction='horizontal')
		combo_2 = append_images(back, direction='horizontal')
		combined = append_images([combo_1, combo_2], direction='vertical')
		#image_tensor = self.transform(combined)
		image_tensor = torch.stack([self.transform(combined)])

		data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
		corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
		categories = data_entries.category_id.to_numpy()

		ego_path = os.path.join(sample_path, 'ego.png')
		ego_image = Image.open(ego_path)
		ego_image = torchvision.transforms.functional.to_tensor(ego_image)
		road_image = convert_map_to_road_map(ego_image)

		target = {}
		target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
		target['category'] = torch.as_tensor(categories)

		if self.extra_info:
			actions = data_entries.action_id.to_numpy()
			# You can change the binary_lane to False to get a lane with 
			lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)

			extra = {}
			extra['action'] = torch.as_tensor(actions)
			extra['ego_image'] = ego_image
			extra['lane_image'] = lane_image

			return image_tensor, target, road_image, extra
		else:
			return image_tensor, target, road_image




