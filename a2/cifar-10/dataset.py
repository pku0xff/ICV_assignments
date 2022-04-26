'''
JOB LOG:
重点学习了 transforms 方法的使用
'''
import os
import os.path

import cv2
import numpy as np
import pickle
import torch

# # use below packages as you like
import torchvision.transforms as tfs


# from PIL import Image
# import cv2

class CIFAR10(torch.utils.data.Dataset):
    """
        modified from `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    """

    def __init__(self, train=True):
        super(CIFAR10, self).__init__()

        self.base_folder = '../datasets/cifar-10-batches-py'
        self.train_list = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        self.test_list = ['test_batch']

        self.meta = {
            'filename': 'batches.meta',
            'key': 'label_names'
        }

        self.train = train  # training set or test set
        if self.train:
            file_list = self.train_list
        else:
            file_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in file_list:
            file_path = os.path.join(self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()
        self.transform = tfs.Compose([self.flip, self.brightness, self.original])

    def _load_meta(self):
        path = os.path.join(self.base_folder, self.meta['filename'])
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = img.astype(np.float32)
        img = img.transpose(2, 0, 1)

        # https://stackoverflow.com/questions/51677788/data-augmentation-in-pytorch
        # img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data)

    def flip(self, img):
        img = img.transpose(1, 2, 0).astype(np.uint8)
        aug = cv2.flip(img, 90)
        aug = aug.transpose(2, 0, 1).astype(np.float32)
        return aug

    def brightness(self, img):
        img = img.transpose(1, 2, 0).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float64)
        hsv[:, :, 2] = (hsv[:, :, 2] * 1.5).clip(0, 255)
        hsv = hsv.astype(np.uint8)
        aug = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        aug = aug.transpose(2, 0, 1).astype(np.float32)
        return aug

    def original(self, img):
        return img


if __name__ == '__main__':
    from PIL import Image

    # --------------------------------
    # The resolution of CIFAR-10 is tooooo low
    # You can use Lenna.png as an example to visualize and check your code.
    # Submit the origin image "Lenna.png" as well as at least two augmented images of Lenna named "Lenna_aug1.png", "Lenna_aug2.png" ...
    # --------------------------------

    # # Visualize CIFAR-10. For someone who are intersted.
    # train_dataset = CIFAR10()
    # i = 0
    # for imgs, labels in train_dataset:
    #     imgs = imgs.transpose(1,2,0)
    #     cv2.imwrite(f'aug1_{i}.png', imgs)
    #     i += 1
    #     if i == 10:
    #         break 

    # Visualize and save for submission
    img = Image.open('Lenna.png')
    img.save('../results/Lenna.png')

    # 1.Flip
    img = np.asarray(img)
    aug1 = cv2.flip(img, 90)
    aug1 = Image.fromarray(aug1)
    aug1.save(f'../results/Lenna_aug1.png')

    # 2.Brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)
    hsv[:, :, 2] = (hsv[:, :, 2] * 1.5).clip(0, 255)
    hsv = hsv.astype(np.uint8)
    aug2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    aug2 = Image.fromarray(aug2)
    aug2.save(f'../results/Lenna_aug2.png')
