from __future__ import print_function
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import ShapeNetClassficationDataset
from model import PointNetCls256D
import torch.nn.functional as F
from tqdm import tqdm
from utils import log_writter, setting

if __name__ == '__main__':
    opt = setting()

    writter = log_writter(opt.outf, "cls_256D")

    blue = lambda x: '\033[94m' + x + '\033[0m'

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    # if opt.dataset_type == 'shapenet':
    dataset = ShapeNetClassficationDataset(
        root=opt.dataset,
        npoints=opt.num_points)

    test_dataset = ShapeNetClassficationDataset(
        root=opt.dataset,
        split='test',
        npoints=opt.num_points,
        with_data_augmentation=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    num_classes = len(dataset.classes)
    print('classes', num_classes)

    classifier = PointNetCls256D(k=num_classes)

    optimizer = optim.Adam(classifier.parameters(), lr=0.1, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # classifier.cuda()

    num_batch = len(dataset) / opt.batchSize

    count = 0
    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points, target
            this_batch_size = points.size()[0]
            optimizer.zero_grad()
            classifier = classifier.train()
            pred = classifier(points)
            loss = F.nll_loss(pred, target)
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (
                epoch, i, num_batch, loss.item(), correct.item() / float(this_batch_size)))
            writter.add_train_scalar("Loss", loss.item(), count)
            writter.add_train_scalar("Acc", correct.item() / float(this_batch_size), count)

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points, target
                classifier = classifier.eval()
                pred = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (
                    epoch, i, num_batch, blue('test'), loss.item(), correct.item() / float(this_batch_size)))

                writter.add_test_scalar("Loss", loss.item(), count)
                writter.add_test_scalar("Acc", correct.item() / float(this_batch_size), count)
            count += 1

        # torch.save(classifier.state_dict(), opt.outf+'/cls_model'+str(opt.layer3_dim)+"_"+str(epoch)+".pth")
        scheduler.step()

    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points, target
        classifier = classifier.eval()
        pred = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))
