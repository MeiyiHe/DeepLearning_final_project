
import torch
from torch import nn
import utils

import torch.nn.functional as F
import numpy as np




class SimpleModel(nn.Module):
	def __init__(self):
		super(SimpleModel, self).__init__()
		self.encoder = torchvision.models.resnet18()
		self.encoder.fc = nn.Identity()

		self.regression = nn.Sequential(OrderedDict([
			('linear1', nn.Linear(512, 2)),
			]))

	def forward(self, x):
		x = self.encoder(x)
		return self.regression(x)

		