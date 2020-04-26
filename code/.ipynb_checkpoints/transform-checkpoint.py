import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import torch



def NormalizeTransform():
	return torchvision.transforms.Compose([
		torchvision.transforms.Normalize(
		mean=[.485, .456, .406],
		std=[.229, .224, .225]),

		transforms.ToTensor(),
		])



def IdentityTransform():
	return torchvision.transforms.ToTensor()