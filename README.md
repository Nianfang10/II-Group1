# **Image Interpretaion - Group1**
Code for Image Interpretation at ETH Zurich FS21.  
Team member: **Liyuan Zhu, Rushan Wang, Nianfang Shi**  

___
The goal is to implement and train a DNN to classify the pixels on remote sensing images.  

First we use a traditional classification method [to be determined], the performance of it:
| Method      | Validate Acc. |Test Acc.|
| ----------- | ----------- | --------- |
| K-means     |             |           |

The architecture of our deep neural network:  
(1)Model 1: A simple CNN architecture.  
![image](https://github.com/Nianfang10/II-Group1/blob/main/visualization/Simple_CNN.png)

(2)Model 2: Simplified UNet.  
![image](https://github.com/Nianfang10/II-Group1/blob/main/visualization/Simplified_UNet.png)
The accuracy of the network:  
| DNN         | Validate Acc. |Test Acc.| Precision |Recall  |  F1  |
| ----------- | ----------- | --------- |-----------|--------|----- |
| Model 1     | 93.46%      |  94.54%   |  87.13%   | 95.50% |90.25%|
| UNet        | 95.99%      |  97.23%   |    93.98% |96.02%  |94.96%|
