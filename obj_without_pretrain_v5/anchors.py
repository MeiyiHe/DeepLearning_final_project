from boxes import box_iou, nms
from helper import compute_ats_bounding_boxes

import torch 
import matplotlib.patches as patches
import matplotlib.pyplot as plt 



# calculate the offset between actual coordinates and anchor boxes
def get_offsets(anchor_boxes, actual_boxes):
    actual_width = actual_boxes[:, 2] - actual_boxes[:, 0]
    actual_height = actual_boxes[:, 3] - actual_boxes[:, 1]
    actual_center_x = actual_boxes[:, 0] + 0.5*actual_width
    actual_center_y = actual_boxes[:, 1] + 0.5*actual_height

    gt_width = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    gt_height = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    gt_center_x = anchor_boxes[:, 0] + 0.5*gt_width
    gt_center_y = anchor_boxes[:, 1] + 0.5*gt_height

    delta_x = (gt_center_x - actual_center_x) / actual_width
    delta_y = (gt_center_y - actual_center_y) / actual_height
    delta_scaleX = torch.log(gt_width / actual_width)
    delta_scaleY = torch.log(gt_height / actual_height)

    offsets = torch.cat([delta_x.unsqueeze(0), 
                    delta_y.unsqueeze(0),
                    delta_scaleX.unsqueeze(0),
                    delta_scaleY.unsqueeze(0)],
                dim=0)
    return offsets.permute(1,0)


def get_bbox_gt(bboxes1, classes, anchor_boxes, sz, device):
    # ex1, ey1, ex2, ey2 are the four coordinates of fl, br
    # bbox in meters
    # classes not used
    bboxes = bboxes1.clone().to(device)
    actual_boxes = []
    for box in bboxes:
        # convert to pixel format
        actual_box = convert_box_format(box)
        actual_boxes.append(actual_box)

    actual_boxes = torch.stack(actual_boxes).to(device)
    high_threshold = 0.7
    low_threshold = 0.3

    actual_width = actual_boxes[:, 2] - actual_boxes[:, 0]
    actual_height = actual_boxes[:, 3] - actual_boxes[:, 1]
    actual_center_x = actual_boxes[:, 0] + 0.5*actual_width
    actual_center_y = actual_boxes[:, 1] + 0.5*actual_height
    
    gt_widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    gt_heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    gt_center_x = anchor_boxes[:, 0] + 0.5*gt_widths
    gt_center_y = anchor_boxes[:, 1] + 0.5*gt_heights

    ious = box_iou(anchor_boxes, actual_boxes)
    ious2 = box_iou(actual_boxes, anchor_boxes)
    vals, inds = torch.max(ious, dim=1)
    vals2, inds2 = torch.max(ious2, dim=1)

    shape_i = anchor_boxes.shape[0]
    gt_classes = torch.zeros((shape_i)).type(torch.long).to(device)
    gt_offsets = torch.zeros((shape_i, 4)).type(torch.double).to(device)
    
    gt_classes[vals > high_threshold] = 1
    gt_classes[vals < low_threshold] = 0 # background anchors
    gt_classes[(vals >= low_threshold) & (vals < high_threshold)] = -1 # anchors to ignore

    k = 0

    for box, val, ind in zip(actual_boxes, vals2, inds2):
        if val.item() < high_threshold:
            k += 1
            gt_classes[ind] = 1
            offset = get_offsets(anchor_boxes[ind].unsqueeze(0), box.unsqueeze(0)).squeeze(0)
            gt_offsets[ind] = offset

    actual_box_es = actual_boxes[inds[vals > high_threshold]]
    ref_boxes = anchor_boxes[vals > high_threshold]
    g_offsets = get_offsets(ref_boxes, actual_box_es)
    gt_offsets[vals > high_threshold] = g_offsets

    return gt_classes, gt_offsets


def batched_coor_threat_updated(ite, predicted_offsets, anchor_boxes, target, gt_classes, batch_sz, nms_threshold=0.1, plot=False):
  # predicted offsets, target_offests, coor_in_meter
    batch_coor = []
    batched_threat_sum=0
    original_anchor = anchor_boxes.clone()
    original_predicted_offsets = predicted_offsets.clone()
    original_gt_classes = gt_classes.clone()

    for i in range(batch_sz):
        anchor_boxes = original_anchor
        predicted_offsets = original_predicted_offsets[i]
        if i == 0:
          cur_target = torch.from_numpy(target[0])
        else:
          cur_target = torch.from_numpy(target[1])
    
        gt_classes = original_gt_classes[i]

        inds = (gt_classes != 0)
        anchor_boxes = anchor_boxes[inds]
        predicted_offsets = predicted_offsets[inds]
        gt_classes = gt_classes[inds]


        delta_x = predicted_offsets[:,0]
        delta_y = predicted_offsets[:,1]
        delta_scaleX = predicted_offsets[:,2]
        delta_scaleY = predicted_offsets[:,3]

        gt_widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
        gt_heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]
        gt_center_x = anchor_boxes[:, 0] + 0.5 * gt_widths
        gt_center_y = anchor_boxes[:, 1] + 0.5 * gt_heights

        ex_width = gt_widths / torch.exp(delta_scaleX)
        ex_height = gt_heights / torch.exp(delta_scaleY)
        ex_center_x = gt_center_x - delta_x*ex_width
        ex_center_y = gt_center_y - delta_y*ex_height

        ex1 = ex_center_x - 0.5*ex_width
        ex2 = ex_center_x + 0.5*ex_width
        ey1 = ex_center_y - 0.5*ex_height
        ey2 = ex_center_y + 0.5*ex_height


        pred_boxes = torch.cat([ex1.unsqueeze(0), ey1.unsqueeze(0), ex2.unsqueeze(0), ey2.unsqueeze(0)], dim=0).permute(1,0)
        pred_boxes = pred_boxes.type(torch.float32)
        gt_classes = gt_classes.type(torch.float32)
        cur_target = cur_target.type(torch.float32)
   
        inds = nms(pred_boxes, gt_classes, nms_threshold)
        pred_boxes = pred_boxes[inds]
        coordinate_list = []

        for box in pred_boxes:
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x1 = (x1-400)/10
            x2 = (x2-400)/10
            y1 = (y1-400)/-10
            y2 = (y2-400)/-10
            width = abs(x1 - x2)
            height = abs(y1 - y2)
            coordinate_list.append(torch.tensor([x2, x2, x1, x1, y2, y1, y2, y1]).view(-1, 4))
            
        coordinate_list = torch.stack(coordinate_list)
        batched_threat_sum += compute_ats_bounding_boxes(coordinate_list, cur_target)
        batched_threat_sum /= batch_sz
        batch_coor.append(coordinate_list)
        #visActual(cur_target, ite,i)
    
    return batch_coor, batched_threat_sum



def get_coordinate(predicted_offsets, anchor_boxes, target, gt_classes, nms_threshold=0.1, plot=False):

    #cur_target = torch.from_numpy(target)
    #gt_classes = original_gt_classes

    inds = (gt_classes != 0)
    anchor_boxes = anchor_boxes[inds]
    predicted_offsets = predicted_offsets[inds]
    gt_classes = gt_classes[inds]


    delta_x = predicted_offsets[:,0]
    delta_y = predicted_offsets[:,1]
    delta_scaleX = predicted_offsets[:,2]
    delta_scaleY = predicted_offsets[:,3]

    gt_widths = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    gt_heights = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    gt_center_x = anchor_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = anchor_boxes[:, 1] + 0.5 * gt_heights

    ex_width = gt_widths / torch.exp(delta_scaleX)
    ex_height = gt_heights / torch.exp(delta_scaleY)
    ex_center_x = gt_center_x - delta_x*ex_width
    ex_center_y = gt_center_y - delta_y*ex_height

    ex1 = ex_center_x - 0.5*ex_width
    ex2 = ex_center_x + 0.5*ex_width
    ey1 = ex_center_y - 0.5*ex_height
    ey2 = ex_center_y + 0.5*ex_height


    pred_boxes = torch.cat([ex1.unsqueeze(0), ey1.unsqueeze(0), ex2.unsqueeze(0), ey2.unsqueeze(0)], dim=0).permute(1,0)
    pred_boxes = pred_boxes.type(torch.float32)
    gt_classes = gt_classes.type(torch.float32)
    cur_target = cur_target.type(torch.float32)

    inds = nms(pred_boxes, gt_classes, nms_threshold)
    pred_boxes = pred_boxes[inds]
    coordinate_list = []

    for box in pred_boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        x1 = (x1-400)/10
        x2 = (x2-400)/10
        y1 = (y1-400)/-10
        y2 = (y2-400)/-10
        width = abs(x1 - x2)
        height = abs(y1 - y2)
        coordinate_list.append(torch.tensor([x2, x2, x1, x1, y2, y1, y2, y1]).view(-1, 4))    
    coordinate_list = torch.stack(coordinate_list)
    return coordinate_list


# convert to 800*800
def convert_box_format(box):

    point_squence = torch.stack([box[:, 0], box[:, 1], box[:, 3], box[:, 2], box[:, 0]]).T
    point_squence[0] *= 10
    point_squence[0] += 400

    point_squence[1] = -point_squence[1] * 10  + 400

    fr_l_x = point_squence[0][0]
    fr_r_x = point_squence[0][1]
    bk_r_x = point_squence[0][2]
    bk_l_x = point_squence[0][3]

    fr_l_y = point_squence[1][0]
    fr_r_y = point_squence[1][1]
    bk_r_y = point_squence[1][2]
    bk_l_y = point_squence[1][3]

    ex1 = min(fr_l_x, fr_r_x, bk_r_x, bk_l_x)
    ex2 = max(fr_l_x, fr_r_x, bk_r_x, bk_l_x)
    ey1 = min(fr_l_y, fr_r_y, bk_r_y, bk_l_y)
    ey2 = max(fr_l_y, fr_r_y, bk_r_y, bk_l_y)

    return torch.tensor([ex1, ey1, ex2, ey2]).view(4)