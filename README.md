[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <img alt="NumPy" src="https://img.shields.io/badge/numpy%20-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white" />  [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FMr-TalhaIlyas%2FConfusion_Matrix_for_Objecti_Detection_Models&count_bg=%233DC8C7&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Confusion Matrix for Object Detection Models

This repo contains script for creating confusion matrix for object detection models, provided  ```.txt``` files as shown in sample data dir. This repos is adapted and modified form following repos,

* [repo_1 kaanakan](https://github.com/kaanakan/object_detection_confusion_matrix)
* [repo_2 pythorch vision](https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py)
* [repo_3 whynotw](https://github.com/whynotw/YOLO_metric)
* [repo_4 tf tenosrboard](https://www.tensorflow.org/tensorboard/image_summaries)

## Dependencies
Repo only require some basic libs. like,
1. Numpy 
2. Matplotlib

## Make Your own Confusion Matrix

After you train you abject detection model and get the final ouputs after non-max-suppression(NMS), just convert all the predictions to format as in **sample_data** dir in repo.
After saving the predictionsin their respective dir. in .txt format just run change the paths and run the script.
The text files are in following format, (In sample data I have 4 classes namely [ "blossom_end_rot", "graymold","powdery_mildew","spider_mite","spotting_disease"])
#### GT
```
spider_mite 133 512 203 598
spider_mite 216 186 336 386
```
This will be converted to a 2x5 array, format given below

```
# class idx, 4 coord
3 133 512 203 598
3 216 186 336 386
```

The first one is class name following digits are bounding box coorinated format:(x_min, y_min, x_max, y_max)
#### Pred
```
spider_mite 0.99998844 220 196 341 412
spider_mite 0.98192304 0 294 32 505
```
The first one is class name, second one is confidence score and following digits are bounding box coorinated format:(x_min, y_min, x_max, y_max).
This will be converted to a 2x6 array, format given below
```
# 4 coord, confd, class idx
220 196 341 412 0.99998844 3
0 294 32 505    0.98192304 3
```

## Notes

1. change the class name in the script, i have use pascal-voc classes.
2. ```plot_confusion_matrix``` function has 3 modes 

* normalize = True -> to normalize the values between 0 and 1 in the plot
* show_text = True -> to show text in confucion matrix blocks or not
* show_fpfn = True -> whether to show FP/FN in the final matix

Also you can give this function the ```cm``` matrix obtained via ```sklearn``` lib and a list of class names it'll a plot an image from the given array.
## Usage
```python
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
 
    conf_mat.process_batch(y_p, y_t) 
    
cm = conf_mat.matrix
c_m = plot_confusion_matrix(cm, class_names, normalize = True, show_text = True, show_fpfn = True)
```
## Sample Output

Confucion matrix for pascal_voc dataset, the ouput is from custom trained YOLO_v1 from scratch that's why the results are not so good.

![alt text](https://github.com/Mr-TalhaIlyas/Confusion_Matrix_for_Objecti_Detection_Models/blob/master/images/CM.png?raw=true)
