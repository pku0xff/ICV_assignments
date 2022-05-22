import os
from matplotlib import image
import numpy as np
import torch
import torch.utils.data
import random
import cv2
import math
from utils import plot_save_dataset


class SingleShapeDataset(torch.utils.data.Dataset):
    def __init__(self, size):

        self.w = 128
        self.h = 128
        self.size = size
        print("size", self.size)

    def _draw_shape(self, img, mask, shape_id):
        buffer = 20
        y = random.randint(buffer, self.h - buffer - 1)
        x = random.randint(buffer, self.w - buffer - 1)
        s = random.randint(buffer, self.h // 4)
        color = tuple([random.randint(0, 255) for _ in range(3)])

        if shape_id == 1:
            cv2.rectangle(mask, (x - s, y - s), (x + s, y + s), 1, -1)
            cv2.rectangle(img, (x - s, y - s), (x + s, y + s), color, -1)

        elif shape_id == 2:
            cv2.circle(mask, (x, y), s, 1, -1)
            cv2.circle(img, (x, y), s, color, -1)

        elif shape_id == 3:
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            cv2.fillPoly(mask, points, 1)
            cv2.fillPoly(img, points, color)

    def __getitem__(self, idx):
        np.random.seed(idx)

        n_class = 1
        masks = np.zeros((n_class, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[..., :] = np.asarray([random.randint(0, 255) for _ in range(3)])[None, None, :]

        obj_ids = np.zeros((n_class))

        shape_code = random.randint(1, 3)
        self._draw_shape(img, masks[0, :], shape_code)
        obj_ids[0] = shape_code

        boxes = np.zeros((n_class, 4))
        pos = np.where(masks[0])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        boxes[0, :] = np.asarray([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)

        return img, target

    def __len__(self):
        return self.size


# ----------TODO------------
# Implement ShapeDataset.
# Refer to `SingleShapeDataset` for the shape parameters 
# ----------TODO------------


class ShapeDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.w = 128
        self.h = 128
        self.size = size
        print("size", self.size)

    # the same as SingleShapeDataset
    def _draw_to_mask(self, x, y, s, color, mask, shape_id):
        if shape_id == 1:
            cv2.rectangle(mask, (x - s, y - s), (x + s, y + s), color, -1)

        elif shape_id == 2:
            cv2.circle(mask, (x, y), s, color, -1)

        elif shape_id == 3:
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            cv2.fillPoly(mask, points, color)

    def _draw_to_img(self, x, y, s, img, shape_id):
        color = tuple([random.randint(0, 255) for _ in range(3)])
        if shape_id == 1:
            cv2.rectangle(img, (x - s, y - s), (x + s, y + s), color, -1)
        elif shape_id == 2:
            cv2.circle(img, (x, y), s, color, -1)
        elif shape_id == 3:
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            cv2.fillPoly(img, points, color)

    def __getitem__(self, idx):
        np.random.seed(idx)
        n_shapes = random.randint(1, 3)

        image_id = torch.tensor([idx])  # ???
        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # 和single一样
        bg_color = [random.randint(0, 255) for _ in range(3)]
        masks = np.zeros((n_shapes, self.h, self.w))
        img = np.zeros((self.h, self.w, 3))
        img[..., :] = np.asarray(bg_color)[None, None, :]

        obj_ids = np.zeros((n_shapes))

        def overlap_area(array1, array2):
            xmin1, ymin1, xmax1, ymax1 = array1.astype(np.uint8)
            xmin2, ymin2, xmax2, ymax2 = array2.astype(np.uint8)
            box1 = np.zeros((128, 128))
            box2 = np.zeros((128, 128))
            box1[xmin1:xmax1, ymin1:ymax1] = 1
            box2[xmin2:xmax2, ymin2:ymax2] = 1
            area1 = np.sum(box1)
            area2 = np.sum(box2)
            overlap = np.sum(np.where(box1 + box2 > 1, 1, 0))
            return area1, area2, overlap

        boxes = np.zeros((n_shapes, 4))
        # 在single的基础上套个循环
        # NMS策略: 两个boxes之间的的重合度高于50%则重新生成
        for i in range(n_shapes):
            while True:
                masks[i] = 0
                shape_code = random.randint(1, 3)
                y = random.randint(20, self.h - 20 - 1)
                x = random.randint(20, self.w - 20 - 1)
                s = random.randint(20, self.h // 4)
                self._draw_to_mask(x, y, s, 1, masks[i, :], shape_code)
                obj_ids[i] = shape_code
                pos = np.where(masks[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes[i, :] = np.asarray([xmin, ymin, xmax, ymax])
                flag = 0
                for j in range(i):
                    a1, a2, overlap = overlap_area(boxes[i], boxes[j])
                    if overlap > 0.5 * a1 or overlap > 0.5 * a2:
                        break  # 存在一个已有图形j，与当前图形i重合面积过大
                    else:
                        flag += 1
                if flag == i:
                    # 符合条件overlap<50%，将图形加入最终的img里
                    self._draw_to_img(x, y, s, img, shape_code)
                    # 覆盖之前生成的mask
                    for j in range(i):
                        self._draw_to_mask(x, y, s, 0, masks[j], shape_code)
                    break

        # 和single一样
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        masks = torch.flip(masks, [0])

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)

        return img, target

    def __len__(self):
        return self.size


if __name__ == '__main__':
    # dataset = SingleShapeDataset(10)
    dataset = ShapeDataset(10)
    path = "results/"

    for i in range(10):
        imgs, labels = dataset[i]
        # print(labels)
        plot_save_dataset(path + str(i) + "_data.png", imgs, labels)
