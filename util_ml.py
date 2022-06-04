# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 08:54:08 2022

@author: yanxi
"""
import time
import torch
import numpy as np


def train(model, train_loader, criterion, optimizer, n_epochs=10):
    # model in training mode
    model.train()
    for epoch in range(1, n_epochs+1):

        train_loss = 0.0
        t = time.time()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader)
        t = time.time() - t

        print('Epoch: {} \tTraining Loss: {:.6f}, Time: {:.2f}'.format(
            epoch, train_loss, t))
    
    # model in evaluation mode
    model.eval()


def test(model, test_loader, criterion):
    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    # model in evaluation mode
    model.eval()
    t = time.time()
    for data, target in test_loader:
        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    t = time.time() - t
    
    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader)
    print(f'Test Loss: {test_loss:.6f}, Time: {t:.6f}\n')

    for label in range(10):
        print(
            f'Test Accuracy of {label}: {100 * class_correct[label] / class_total[label]:.2f}% '
            f'({int(np.sum(class_correct[label]))}/{int(np.sum(class_total[label]))})'
        )

    print(
        f'\nTest Accuracy (Overall): {100 * np.sum(class_correct) / np.sum(class_total):.2f}% ' 
        f'({int(np.sum(class_correct))}/{int(np.sum(class_total))})'
    )

