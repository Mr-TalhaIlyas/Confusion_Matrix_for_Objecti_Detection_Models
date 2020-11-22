from tqdm import tqdm
import xmltodict
import numpy as np
import glob
import os, copy
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

# https://github.com/kaanakan/object_detection_confusion_matrix
# https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
# https://github.com/whynotw/YOLO_metric
# https://www.tensorflow.org/tensorboard/image_summaries
'''
This code is adapted from abvoe links to plot a confusion matrix  for object detection
model for details kinly visit them

Kindly read README.md for details.
'''
    
def box_iou_calc(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min = 0, a_max = None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(self, num_classes, CONF_THRESHOLD = 0.3, IOU_THRESHOLD = 0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
    
    def process_batch(self, detections, labels):
        '''
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        '''
        detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        gt_classes = labels[:, 0].astype(np.int16)
        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = []
        for i in range(want_idx[0].shape[0]):
            all_matches.append([want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]])
        
        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0: # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index = True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index = True)[1]]


        for i, label in enumerate(labels):
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                gt_class = gt_classes[i]
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[(gt_class), detection_class] += 1
            else:
                gt_class = gt_classes[i]
                self.matrix[(gt_class), self.num_classes] += 1
        
        for i, detection in enumerate(detections):
            if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                detection_class = detection_classes[i]
                self.matrix[self.num_classes ,detection_class] += 1
            

def plot_confusion_matrix(cm, class_names, normalize = True, show_text = True, show_fpfn = False):
    '''
    Parameters
    ----------
    cm : a nxn dim numpy array.
    class_names: a list of class names (str type)
    normalize: whether to normalize the values
    show_text: whether to show value in each block of the matrix, If matrix is large like 10x10 or 20x20 it's better to set it to false
               because it'll be difficult to read values but you can see the network behaviour via color map.
    show_fpfn: whether to show false positives on GT axis and false negatives on Pred axis. FN -> not detected & FP -> wrong detections
    Returns
    -------
    fig: a plot of confusion matrix along with colorbar
    '''
    if show_fpfn:
        conf_mat = cm
        x_labels = copy.deepcopy(class_names)
        y_labels = copy.deepcopy(class_names)
        x_labels.append('FN')
        y_labels.append('FP')
    else:
        conf_mat = cm[0:cm.shape[0]-1, 0:cm.shape[0]-1]
        x_labels = class_names
        y_labels = class_names
    my_cmap = 'Greens'# viridis, seismic, gray, ocean, CMRmap, RdYlBu, rainbow, jet, Blues, Greens, Purples
    
    
    c_m = conf_mat
    
    if normalize:
        row_sums = c_m.sum(axis=1)
        c_m = c_m / row_sums[:, np.newaxis]
        c_m = np.round(c_m, 3)
    
    print('*'*80)
    print('NOTE: In confusion_matrix the last coloumn "FP/FN" shows False Positives in Groundtruths \
          \nand False Negatives in Predictions')
    print('*'*80)
    
    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(c_m, cmap = my_cmap) 
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(y_labels)))
    ax.set_yticks(np.arange(len(x_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(y_labels)
    ax.set_yticklabels(x_labels)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center", rotation_mode="anchor")#ha=right
    
    # Loop over data dimensions and create text annotations.
    def clr_select(i, j):
        if i==j:
            color="green"
        else:
            color="red"
        return color
    if show_text:
        for i in range(len(x_labels)):
            for j in range(len(y_labels)):
                text = ax.text(j, i, c_m[i, j], color="k", ha="center", va="center")#color=clr_select(i, j)
    
    ax.set_title("Normalized Confusion Matrix")
    fig.tight_layout()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    plt.colorbar(sm)
    plt.show() 
    return fig     

def read_txt(txt_file, pred=True):
    '''
    Parameters
    ----------
    txt_file : txt file path to read
    pred : if your are raedinf prediction txt file than it'll have 5 values 
    (i.e. including confdience) whereas GT won't have confd value. So set it
    to False for GT file. The default is True.
    Returns
    -------
    info : a list haing 
        if pred=True => detected_class, confd, x_min, y_min, x_max, y_max
        if pred=False => detected_class, x_min, y_min, x_max, y_max
    '''
    x = []
    with open(txt_file, 'r') as f:
        info = []
        x = x + f.readlines()
        for item in x:
            item = item.replace("\n", "").split(" ")
            if pred == True:
                # for preds because 2nd value in preds is confidence
                det_class = item[0]
                confd = float(item[1])
                x_min = int(item[2])
                y_min = int(item[3])
                x_max = int(item[4])
                y_max = int(item[5])
                
                info.append((x_min, y_min, x_max, y_max, confd, det_class))
                            
            else:
                # for preds because 2nd value in preds is confidence
                det_class = item[0]
                x_min = int(item[1])
                y_min = int(item[2])
                x_max = int(item[3])
                y_max = int(item[4])
                
                info.append((det_class, x_min, y_min, x_max, y_max))
                
        return info
    
def IoU(target_boxes , pred_boxes):
    xA = np.maximum( target_boxes[ ... , 0], pred_boxes[ ... , 0] )
    yA = np.maximum( target_boxes[ ... , 1], pred_boxes[ ... , 1] )
    xB = np.minimum( target_boxes[ ... , 2], pred_boxes[ ... , 2] )
    yB = np.minimum( target_boxes[ ... , 3], pred_boxes[ ... , 3] )
    interArea = np.maximum(0.0, xB - xA ) * np.maximum(0.0, yB - yA )
    boxAArea = (target_boxes[ ... , 2] - target_boxes[ ... , 0]) * (target_boxes[ ... , 3] - target_boxes[ ... , 1])
    boxBArea = (pred_boxes[ ... , 2] - pred_boxes[ ... , 0]) * (pred_boxes[ ... , 3] - pred_boxes[ ... , 1])
    iou = interArea / ( boxAArea + boxBArea - interArea )
    iou = np.nan_to_num(iou)
    return iou    

def process(x, class_names, gt = True):
    '''
    Parameters
    ----------
    x : class_name, x_min, y_min, x_max, y_max
    Returns
    -------
    x : class_index, x_min, y_min, x_max, y_max
    '''
    if gt:
        clas = x[:,0]
        temp = []
        for i in range(len(clas)):
            temp.append(class_names.index(clas[i]))
        temp = np.array(temp)
        x[:,0] = temp
        x = x.astype(np.int32)
    else:
        clas = x[:,-1]
        temp = []
        for i in range(len(clas)):
            temp.append(class_names.index(clas[i]))
        temp = np.array(temp)
        x[:,-1] = temp
        x = x.astype(np.float32)
    
    return x

class_names = ['aeroplane', 'bicycle', 'bird','boat','bottle', 'bus', 'car',
                'cat', 'chair', 'cow','diningtable','dog', 'horse', 'motorbike',
                'person','pottedplant','sheep', 'sofa', 'train', 'tvmonitor']
#class_names = [ "blossom_end_rot", "graymold","powdery_mildew","spider_mite","spotting_disease"]

gt = glob.glob(os.path.join('D:/cw_projects/conf_mat/gt/','*.txt'))
pred = glob.glob(os.path.join('D:/cw_projects/conf_mat/pred/','*.txt'))
conf_mat = ConfusionMatrix(num_classes = len(class_names), CONF_THRESHOLD = 0.3, IOU_THRESHOLD = 0.5)

for i in range(len(gt)):
    y_t = np.asarray(read_txt(gt[i], pred=False))
    y_p = np.asarray(read_txt(pred[i], pred=True))
    
    if y_p.size !=0:
        y_t = process(y_t, class_names)
        y_p = process(y_p, class_names, gt = False)
        conf_mat.process_batch(y_p, y_t) 
    
cm = conf_mat.matrix


c_m = plot_confusion_matrix(cm, class_names, normalize = True, show_text = True, show_fpfn = True)
