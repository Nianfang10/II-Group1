# **Image Interpretaion - Group1**
Code for Image Interpretation Lab 1 at ETH Zurich FS21.  
Team member: **Liyuan Zhu, Rushan Wang, Nianfang Shi**  

The goal of Lab 1 is to implement and train a DNN to classify the pixels on remote sensing images.  

We have tried three different architecture: A simple CNN, UNet and SegNet.
___
**Architecture 1: A simple CNN**  
![image](https://github.com/Nianfang10/II-Group1/blob/main/visualization/Simple_CNN.png)
___
**Architecture 2: Simplified UNet**  
![image](https://github.com/Nianfang10/II-Group1/blob/main/visualization/Simplified_UNet.png)  
___
**Architecture 3: SegNet**  
![image](https://github.com/Nianfang10/II-Group1/blob/main/visualization/Simplified_SegNet.png)  
___
Performance:  
| DNN         | Validate Acc. |Test Acc.| Precision |Recall  |  F1  |
| ----------- | ----------- | --------- |-----------|--------|----- |
|Simple CNN   | 93.46%      |  94.54%   |  87.13%   | 95.50% |90.25%|
| UNet        | **95.99%**     |  **97.23%**   |    **93.98%** |96.02%  |**94.96%**|
| SegNet      |94.39%       |   96.36%  | 91.26%    | **96.28%** | 93.49%|


___
# **Reference**