from __future__ import print_function
import os
import random
import torch
import torch.nn.parallel
import torch.utils.data
from dataset import ShapeNetClassficationDataset
from model import PointNetCls1024D
import numpy as np
from utils import write_points, setting
import cv2

if __name__ == '__main__':
    opt = setting()
    blue = lambda x: '\033[94m' + x + '\033[0m'
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    test_dataset = ShapeNetClassficationDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        with_data_augmentation=False)

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=5,
        shuffle=True,
        num_workers=int(opt.workers))

    num_classes = len(test_dataset.classes)
    print('classes', num_classes)

    classifier = PointNetCls1024D(k=num_classes)

    # load weights:
    classifier.load_state_dict(torch.load("results/cls_model_1024D.pth"))

    classifier.eval()

    for i, data in enumerate(testdataloader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points, target
        classifier = classifier.eval()

        # ----------TODO------------
        #  Compute the value/cruciality of each point. And visualize the points with colormap.
        #  e.g. cv2.applyColorMap((X.astype(np.uint8), cv2.COLORMAP_JET )
        # ----------TODO------------
        out1 = classifier.get_feature.layer1(points.transpose(1, 2))
        out2 = classifier.get_feature.layer2(out1)
        crucialty = classifier.get_feature.layer3(out2)
        feat_value = torch.max(crucialty, dim=-1).values.detach().numpy() * 255
        color_vis_feat = cv2.applyColorMap(feat_value.astype(np.uint8), cv2.COLORMAP_JET)

        # feat_value is the final value/cruciality of each point. The range is from 0 to 1 and the larger the better.
        # weite_points() save colors in order of #BGR
        points = points.permute(0, 2, 1)
        write_points(os.path.join(opt.outf, str(0) + ".ply"), points.numpy()[0, ...], color_vis_feat[0, ...],
                     feat_value[0, ...])
        write_points(os.path.join(opt.outf, str(1) + ".ply"), points.numpy()[1, ...], color_vis_feat[1, ...],
                     feat_value[1, ...])
        write_points(os.path.join(opt.outf, str(2) + ".ply"), points.numpy()[2, ...], color_vis_feat[2, ...],
                     feat_value[2, ...])
        write_points(os.path.join(opt.outf, str(3) + ".ply"), points.numpy()[3, ...], color_vis_feat[3, ...],
                     feat_value[3, ...])
        write_points(os.path.join(opt.outf, str(4) + ".ply"), points.numpy()[4, ...], color_vis_feat[4, ...],
                     feat_value[4, ...])

        break
