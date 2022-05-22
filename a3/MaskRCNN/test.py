from tkinter import Label
import utils
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNN
from dataset import SingleShapeDataset
from utils import plot_save_output
import torch
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt


def calculate_AP(rank):
    K = rank.shape[0]
    total_object = np.sum(rank)
    recall = np.zeros(K + 1)
    precision = np.zeros(K + 1)
    if total_object == 0:
        return 0
    # 计算原始值
    recall[0] = 0
    precision[0] = 1
    for i in range(K):
        recall[i + 1] = np.sum(rank[0:i + 1]) / total_object
        precision[i + 1] = np.sum(rank[0:i + 1]) / (i + 1)
    # 平滑
    for i in range(K):
        precision[i] = np.max(precision[i:])
    recall_interp = np.arange(11) / 10.0
    precision_interp = np.interp(recall_interp, recall, precision)
    plt.plot(recall, precision, '-x')
    plt.show()
    return np.mean(precision_interp)


# the outputs includes: 'boxes', 'labels', 'masks', 'scores'
# boxes: np.asarray([xmin, ymin, xmax, ymax])
# labels: shape_code
# masks: 整个mask框
# scores: ???
# 计算mAP在这两个任务之间有什么区别？
def compute_segmentation_ap(output_list, gt_labels_list, iou_threshold=0.5):
    N = len(output_list)
    APs = np.zeros(N)

    def calculate_IoU(mask1, mask2):
        # mask in size 128*128
        mask1 = np.where(mask1.numpy() > 0, 1, 0)
        mask2 = np.where(mask2.numpy() > 0, 1, 0)
        I = np.sum(np.where(mask1 + mask2 == 2, 1, 0))
        U = np.sum(np.where(mask1 + mask2 > 0, 1, 0))
        return I / U

    for i in range(N):
        print(f'image {i}:')
        K = len(output_list[i]['masks'])
        # masks size=(K, 1, 128, 128)
        rank = np.zeros(K)  # 1 for true positive, 0 for false prediction
        for j in range(K):
            print(f'prediction {j}:')
            pred_mask = output_list[i]['masks'][j]
            gt_mask = gt_labels_list[i]['masks'][0]
            pred_label = output_list[i]['labels'][j]
            gt_label = gt_labels_list[i]['labels'][0]
            iou = calculate_IoU(pred_mask, gt_mask)
            print(f'iou: {iou}')
            print(f'pred label {pred_label}, gt label {gt_label}')
            if iou > iou_threshold and pred_label == gt_label:
                rank[j] = 1  # 表示score第j高的结果是正例
        APs[i] = calculate_AP(rank)

    mAP_segmentation = np.mean(APs)
    print('mAP_segmentation =', mAP_segmentation)
    return mAP_segmentation


# a prediction is positive if IoU ≥ 0.5
def compute_detection_ap(output_list, gt_labels_list, iou_threshold=0.5):
    N = len(output_list)
    APs = np.zeros(N)

    def calculate_IoU(box1, box2):
        xmin1, ymin1, xmax1, ymax1 = box1.numpy().astype(np.uint8)
        xmin2, ymin2, xmax2, ymax2 = box2.numpy().astype(np.uint8)
        b1 = np.zeros((128, 128))
        b2 = np.zeros((128, 128))
        b1[xmin1:xmax1, ymin1:ymax1] = 1
        b2[xmin2:xmax2, ymin2:ymax2] = 1
        area1 = np.sum(b1)
        area2 = np.sum(b2)
        overlap = np.sum(np.where(b1 + b2 > 1, 1, 0))
        return overlap / (area1 + area2 - overlap)

    for i in range(N):
        K = len(output_list[i]['masks'])
        rank = np.zeros(K)  # 1 for true positive, 0 for false prediction
        for j in range(K):
            pred_box = output_list[i]['boxes'][j]
            gt_box = gt_labels_list[i]['boxes'][0]
            pred_label = output_list[i]['labels'][j]
            gt_label = gt_labels_list[i]['labels'][0]
            iou = calculate_IoU(pred_box, gt_box)
            if iou > iou_threshold and pred_label == gt_label:
                rank[j] = 1  # 表示score第j高的结果是正例
        APs[i] = calculate_AP(rank)

    mAP_detection = np.mean(APs)
    print('mAP_detection =', mAP_detection)
    return mAP_detection


dataset_test = SingleShapeDataset(10)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)

num_classes = 4

# get the model using the helper function
model = utils.get_instance_segmentation_model(num_classes).double()

device = torch.device('cpu')

# replace the 'cpu' to 'cuda' if you have a gpu
model.load_state_dict(torch.load(r'results/maskrcnn_2.pth', map_location='cpu'))

model.eval()
path = "results/"
# # save visual results
for i in range(10):
    imgs, labels = dataset_test[i]
    output = model([imgs])
    plot_save_output(path + str(i) + "_result.png", imgs, output[0])

# compute AP
gt_labels_list = []
output_label_list = []
with torch.no_grad():
    for i in range(10):
        print(i)
        imgs, labels = dataset_test[i]
        gt_labels_list.append(labels)
        output = model([imgs])
        output_label_list.append(output[0])

mAP_detection = compute_detection_ap(output_label_list, gt_labels_list)
mAP_segmentation = compute_segmentation_ap(output_label_list, gt_labels_list)

np.savetxt(path + "mAP.txt", np.asarray([mAP_detection, mAP_segmentation]))
