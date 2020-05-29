#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2020/2/25 20:22
# @Author ：''
# @FileName: NLP_fizz_buzz.py
import numpy as np
import torch
from torch import nn
from torch import optim


NUM_DIGITS = 10


def fizz_buzz_encode(i):
    if i % 15 == 0: return 3
    elif i % 5 == 0: return 2
    elif i % 3 == 0: return 1
    else: return 0


def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


# Represent each input by an array of its binary digits.
def binary_encode(i, NUM_DIGITS):
    return np.array([i >> d & 1 for d in range(NUM_DIGITS)])


# Define the model
NUM_HIDDEN = 100
model = nn.Sequential(
    nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    nn.ReLU(),
    nn.Linear(NUM_HIDDEN, 4)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)


BATCH_SIZE = 128
trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

# Start training it
for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Find loss on training data
    loss = loss_fn(model(trX), trY).item()
    print("Epoch:", epoch, "Loss:", loss)


# Output now
testX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(1, 101)])

with torch.no_grad():
    testY = model(testX)
predictions = zip(range(1, 101), list(testY.max(1)[1].data.tolist()))

print([fizz_buzz_decode(i, x) for (i, x) in predictions])
print(testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1, 101)]))
print(np.sum(testY.max(1)[1].numpy() == np.array([fizz_buzz_encode(i) for i in range(1, 101)])))
