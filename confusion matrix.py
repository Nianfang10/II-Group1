import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from dataset_windows import SatelliteSet
from networks import CNN_Model


BATCH_SIZE = 16
NUM_WORKERS = 8


if __name__ == '__main__':
    data_pred = []
    data_true = []

    # create the dataset and dataloader
    test_dataset = SatelliteSet(windowsize=128, split='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # create the network
    net = CNN_Model()

    # Load the best checkpoint
    file_path = os.path.join(r'D:\Nianfang\II_Lab\Lab1\checkpoint', f'best-{net.codename}.pth')
    net.load_state_dict(torch.load(file_path)['net'])
    # Put the network in evaluation mode
    net.eval()

    # get the actual and predicted data
    cm_total = np.zeros([3, 3])
    confusion_matrix = np.zeros([3, 3])

    for batch in test_dataloader:
        output = net.forward(batch[0])
        prediction = torch.argmax(output, dim=1).cpu().detach().numpy().astype(int)
        ground_truth = batch[1].data.cpu().numpy().astype(int)
        for i in range(np.shape(prediction)[0]):  # shape of prediction results is the same as batch size, except for the last batch
            for x in range(np.shape(prediction)[1]):
                for y in range(np.shape(prediction)[2]):
                    pred = prediction[i][x, y]
                    true = ground_truth[i][x, y]
                    if true != 99:
                        confusion_matrix[pred, true] += 1


    print(confusion_matrix)
    np.savetxt("confusion matrix.csv", confusion_matrix, delimiter=",", header='background, palm oil trees, clouds')

    # plot
    classes = ('background', 'palm oil trees', 'clouds')  # corresponding to 0, 1, 2, respectively
    plt.figure(figsize=[15, 15])
    plt.imshow(confusion_matrix)
    plt.xticks(np.arange(3), labels=classes)
    plt.xlabel('prediction', fontsize=14)
    plt.yticks(np.arange(3), labels=classes)
    plt.ylabel('ground truth', fontsize=14)
    plt.title("Confusion Matrix", fontsize=24)
    for i in range(3):
        for j in range(3):
            plt.text(i, j, '%d' % (confusion_matrix[j, i]), ha='center', va='center', color='w', fontsize=12.5)
    # plt.show()
    plt.savefig('confusion matrix.jpg')