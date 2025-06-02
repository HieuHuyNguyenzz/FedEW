import numpy as np
from collections import Counter, OrderedDict
from typing import List, Dict
import random
import math
import copy

import torch
from torch.distributions.dirichlet import Dirichlet
from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST, MNIST, CIFAR100
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim import SGD
from sklearn.cluster import OPTICS
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score


seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_entropy(counts: Dict):
    entropy = 0.0
    counts = list(counts.values())
    counts = [0 if value is None else value for value in counts]
    for value in counts:
        entropy += -value/sum(counts) * math.log(value/sum(counts), len(counts)) if value != 0 else 0
    return entropy


def train(net, trainloader, learning_rate: float, proximal_mu: float = None):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=learning_rate)
    net.train()
    running_loss, running_corrects = 0.0, 0
    global_params = copy.deepcopy(net).parameters()
    
    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(images).to(DEVICE)

        if proximal_mu != None:
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term
        else:
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * images.shape[0]
        running_corrects += torch.sum(predicted == labels).item()

    running_loss /= len(trainloader.sampler)
    acccuracy = running_corrects / len(trainloader.sampler)
    return running_loss, acccuracy


def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    corrects, loss = 0, 0.0

    # Khởi tạo các metrics
    precision_metric = MulticlassPrecision(num_classes=10, average='macro').to(DEVICE)
    recall_metric = MulticlassRecall(num_classes=10, average='macro').to(DEVICE)
    f1_metric = MulticlassF1Score(num_classes=10, average='macro').to(DEVICE)

    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images).to(DEVICE)
            predicted = torch.argmax(outputs, dim=1)
            
            # Tính loss
            loss += criterion(outputs, labels).item() * images.shape[0]
            
            # Tính số lượng dự đoán đúng
            corrects += torch.sum(predicted == labels).item()

            # Cập nhật các metrics
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)

    # Tính các giá trị trung bình
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1_score = f1_metric.compute().item()

    loss /= len(testloader.sampler)
    accuracy = corrects / len(testloader.sampler)

    return loss, accuracy, precision, recall, f1_score


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)
   

    
    
