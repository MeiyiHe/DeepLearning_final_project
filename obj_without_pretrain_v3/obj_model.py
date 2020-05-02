from resnet import resnet50,resnet34, resnet18
import torch 
import torch.nn as nn
import torch.nn.functional as F
from fpn import FPN50
from retina import RetinaNet


## import only for testing
import numpy as np
import torchvision
from helper import draw_box, collate_fn
from data_helper import UnlabeledDataset, LabeledDataset
from hrnet import get_seg_model, get_config



class BoundingBox(nn.Module):
    def __init__(self):
        super().__init__()
        # self.input_shape = (800,800)

        # self.encoder = get_seg_model(get_config())
        # self.relu = nn.ReLU(inplace=True)
        # #self.regressor = nn.Conv2d(64, 4*4, kernel_size=1, padding=1, bias=False)
        # self.regressor = nn.Conv2d(1, 4*4, kernel_size=800, padding=0, bias=False)
        # self.pred = nn.Conv2d(1, 4*9, kernel_size=800, padding=0, bias=False)

        ############################################################################
        self.encoder = resnet18()
        self.classifier = nn.Conv2d(512, 64, kernel_size=3, padding=1, bias=False)
        
        self.input_shape = (800,800)
        self.relu = nn.ReLU(inplace=True) 
        self.bn1 = nn.BatchNorm2d(16, momentum=0.01)
        self.classifier1 = nn.Conv2d(3, 18, kernel_size=3, padding=1, bias=False)
        self.regressor = nn.Conv2d(64, 4*4, kernel_size=3, padding=1, bias=False)
        self.pred = nn.Conv2d(64, 4*9, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        #print('input x dimension {}'.format(x.shape))
        x = self.encoder(x)
        #print('after encoder, x dimension {}'.format(x.shape))

        #x = self.classifier(x)
        x = F.interpolate(x, size=self.input_shape, mode='bilinear', align_corners=False)

        #print('after interpolate, x dimension {}'.format(x.shape))
        x = self.relu(self.classifier(x))
        pred_x = self.pred(x)
        #print('pred_x shape {}'.format(pred_x.shape))
        box_x = self.regressor(x)
        #print('box_x shape {}'.format(box_x.shape))

        return pred_x, box_x



if __name__ == '__main__':
    image_folder = '../data'
    annotation_csv = '../data/annotation.csv'
    train_index = np.arange(106,110)
    batch_sz = 2

    transform = torchvision.transforms.ToTensor()
    labeled_trainset = LabeledDataset(image_folder=image_folder,
                                    annotation_file=annotation_csv,
                                    scene_index=train_index,
                                    transform=transform,
                                    extra_info=True
                                   )
    trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=batch_sz, shuffle=True, num_workers=2, collate_fn=collate_fn)

    model = get_seg_model(get_config()).to('cpu')

    for i, (sample, target, road_image, _) in enumerate(trainloader):
        samples = torch.stack(sample).to('cpu')
        #print('samples shape {}'.format(samples.shape))
        samples = samples.view(batch_sz, -1, 256, 306)
        #print('samples shape {}'.format(samples.shape))
        
        #class_target, box_target = get_targets(target, sample)
        out_pred, out_bbox = model(samples)

        break
        out_bbox = out_bbox.view(batch_sz, -1, 4)
        #loss = self.bbox_loss(box_target, class_target, out_bbox)
        train_losses.append(loss.item())


