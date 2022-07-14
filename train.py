# @Time   :2022/3/14 19:11
# @Author :***
# @File   :train
import math
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import logging

from torchvision.datasets import ImageFolder
from tqdm import tqdm
import model.FERVT
from sklearn.metrics import confusion_matrix
from dataset.augment import Albumentations,augment_hsv
from model.FERVT import LabelSmoothingLoss

device = torch.device("cuda")


def get_logger(logger_name, log_file, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    fileHandler = logging.FileHandler(log_file, mode='a')

    logger.setLevel(level)
    logger.addHandler(fileHandler)

    return logger


log1 = get_logger('log1', './model/weight/log.txt', logging.DEBUG)
log2 = get_logger('log2', './model/weight/pltdata.txt', logging.DEBUG)


def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]


class FerDataset(torch.utils.data.Dataset):
    def __init__(self, train=0, one_hot=False, transform=None, augment=False):
        if train==0:
            self.img_path = './dataset/ferplus/FER2013Train'
        elif train==1:
            self.img_path = './dataset/ferplus/FER2013Test'
        elif train==2:
            self.img_path = './dataset/ferplus/FER2013Valid'

        self.transform = transform
        self.img = os.listdir(self.img_path)
        self.onehot = one_hot
        self.augment = augment
        self.albumentations = Albumentations() if augment else None

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_path, self.img[idx]))
        if self.augment:
            image = self.albumentations(image)
            augment_hsv(image)
            if random.random() < 0.5:
                image = np.flipud(image)

            # Flip left-right
            if random.random() < 0.5:
                image = np.fliplr(image)

        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        label = int(self.img[idx].split('_')[0]) - 1
        if self.transform:
            image = Image.fromarray(image)
            image = self.transform(image)

        if self.onehot:
            label = one_hot(label, 8)
        else:
            label = torch.from_numpy(np.array([label]))

        return image, label


def getCKplus(train_transforms):
    dataset = ImageFolder(root = './dataset/CK+/', transform=train_transforms)
    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    return train_dataset, validation_dataset


def getFerplus(train_transforms):
    train_dataset = ImageFolder(root = './dataset/fer2013_plus/Training', transform=train_transforms)
    validation_dataset = ImageFolder(root='./dataset/fer2013_plus/PrivateTest', transform=train_transforms)

    return train_dataset, validation_dataset


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    batch = 0
    avgloss = 0
    with tqdm(dataloader, unit="batch") as epoch:
        for X, y in epoch:
            epoch.set_description(f"Epoch:")
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y.long().squeeze())
            avgloss += loss
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch += 1
            epoch.set_postfix(loss=(avgloss / batch).item(), lr=optimizer.param_groups[0]['lr'])
            if batch % 100 == 0:
                log1.debug(
                    "batch:" + str(
                        batch) + f"    loss: {(avgloss / batch).item():>7f} lr: {optimizer.param_groups[0]['lr']}\n")


def test(dataloader, model, loss_fn,scheduler):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    confusion_pred,confusion_label =[],[]
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y.long().squeeze(1)).item()
            correct += (pred.argmax(1) == one_hot(y.long().squeeze(1), 8).argmax(1).cuda()).type(torch.float).sum().item()
            confusion_pred.append(list(pred.argmax(1).cpu().numpy()))
            confusion_label.append(list(y.squeeze(1).cpu().numpy()))
    scheduler.step()
    test_loss /= num_batches
    correct /= size
    confusion_pred = np.array(sum(confusion_pred,[]))
    confusion_label = np.array(sum(confusion_label,[]))
    log1.debug(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    log1.debug(confusion_matrix(confusion_label, confusion_pred))
    print(confusion_matrix(confusion_label, confusion_pred))
    return 100 * correct, test_loss


def main():
    # dataset load&download
    transform = transforms.Compose([
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
    # training_data,test_data = getCKplus(transform)
    training_data = FerDataset(train=0, transform=transform)
    test_data = FerDataset(train=2, transform=transform)
    train_loader = DataLoader(training_data, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)
    # define network
    FER_VT = model.FERVT.FERVT(device)
    # define loss and optimize
    # loss = LabelSmoothingLoss(8, 0.1)
    loss = torch.nn.CrossEntropyLoss()
    epochs = 100
    optimizer = torch.optim.AdamW(FER_VT.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.2) + 0.2)
    # FER_VT.load_state_dict(torch.load('./model/weight/ck_best.pth'))
    # x.shape(b,3,48,48),y.shape(b,8) if use one hot
    plt_loss = []
    plt_accuracy = []
    best_correct = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------\n")
        log1.debug(f"Epoch {t + 1}\n-------------------------------\n")
        train(train_loader, FER_VT, loss, optimizer)
        correct, test_loss = test(test_loader, FER_VT, loss,scheduler)
        log2.info(str(t)+'___'+str(test_loss)+'___'+str(correct)+'\n')
        # plt_loss.append(test_loss)
        # plt_accuracy.append(correct)
        # if t % 100 == 0:
        #     plt.xlabel('Epoch')
        #     plt.ylabel('accuracy(%)')
        #     plt.plot(np.arange(1, t + 2), np.array(plt_accuracy))
        #     plt.savefig('./dataset/ferplus/accfig/epoch' + str(t) + '_acc.png')
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Loss')
        #     plt.plot(np.arange(1, t + 2), plt_loss)
        #     plt.savefig('./dataset/ferplus/accfig/epoch' + str(t) + '_loss.png')
        if best_correct<correct:
            best_correct = correct
            torch.save(FER_VT.state_dict(), "./model/weight/fer_vt_epoch.pth")
            log1.debug("Saved PyTorch Model State epoch is " + str(t)+"   correct = "+str(correct)+"\n")
            print("Saved PyTorch Model State epoch is " + str(t)+"   correct = "+str(correct)+"\n")
    log1.debug("Done!\n")
    print("Done!\n")


if __name__ == '__main__':
    main()
