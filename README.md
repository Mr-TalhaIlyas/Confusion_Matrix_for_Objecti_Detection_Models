# Confusion Matrix for Object Detection Models

This repe contains script for creating confusion matrix for object detection models this repos is adapted and modified form following repos

* [repo_1](https://github.com/kaanakan/object_detection_confusion_matrix)
* [repo_1](https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py)
* [repo_1](https://github.com/whynotw/YOLO_metric)
* [repo_1](https://www.tensorflow.org/tensorboard/image_summaries)

## Dependencies
Repo only require some basic libs. like,
1. Numpy 
2. Matplotlib

## Make Your own Confusion Matrix

After you train you abject detection model and get the final ouputs after non-max-suppression(NMS), just convert all the predictions to format as in **sample_data** dir in repo.
After saving the predictionsin their respective dir. in .txt format just run change the paths and run the script.
The text files are in following format
#### GT
```
spider_mite 133 512 203 598
spider_mite 216 186 336 386
```

The first one is class name following digits are bounding box coorinated format:(x_min, y_min, x_max, y_max)
#### Pred
```
spider_mite 0.99998844 220 196 341 412
spider_mite 0.98192304 0 294 32 505
```
The first one is class name, second one is confidence score and following digits are bounding box coorinated format:(x_min, y_min, x_max, y_max)

## Notes

1. change the class name in the script, i have use pascal-voc classes.
2. ```plot_confusion_matrix``` function has 3 modes 

* normalize = True -> to normalize the values between 0 and 1 in the plot
* show_text = True -> to show text in confucion matrix blocks or not
* show_fpfn = True -> whether to show FP/FN in the final matix

## Sample Output

Confucion matrix for pascal_voc dataset, the ouput is from custom trained YOLO_v1 from scratch that's why the results are not so good.

![alt text](https://github.com/Mr-TalhaIlyas/Confusion_Matrix_for_Objecti_Detection_Models/blob/master/images/CM.png?raw=true)
