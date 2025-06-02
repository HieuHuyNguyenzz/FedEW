from torchvision import datasets, transforms
import random
import torch
from collections import Counter
from typing import List
from torch.distributions.dirichlet import Dirichlet

seed_value = 42
random.seed(seed_value)


def renormalize(dist: torch.tensor, labels: List[int], label: int):
    idx = labels.index(label)
    dist[idx] = 0
    dist /= sum(dist)
    dist = torch.concat((dist[:idx], dist[idx+1:]))
    return dist

def load_data(dataset_name):
    if dataset_name == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    elif dataset_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    elif dataset_name == "fashion_mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    elif dataset_name == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    return trainset, testset

def partition_data(trainset, num_clients: int, num_iids: int, alpha: float):
    classes = trainset.classes
    client_size = int(len(trainset)/num_clients)
    label_size = int(len(trainset)/len(classes))
    data = list(map(lambda x: (trainset[x][1], x), range(len(trainset))))
    data.sort()
    data = list(map(lambda x: data[x][1], range(len(data))))
    data = [data[i*label_size:(i+1)*label_size] for i in range(len(classes))]

    ids = [[] for _ in range(num_clients)]
    label_dist = []
    labels = list(range(len(classes)))

    for i in range(num_clients):
        if i < num_iids:
            concentration = torch.ones(len(labels)) * 100
        else:
            concentration = torch.ones(len(labels)) * alpha
        dist = Dirichlet(concentration).sample()
        for _ in range(client_size):
            label = random.choices(labels, dist)[0]
            id = random.choices(data[label])[0]
            ids[i].append(id)
            data[label].remove(id)

            if len(data[label]) == 0:
                dist = renormalize(dist, labels, label)
                labels.remove(label)

        counter = Counter(list(map(lambda x: trainset[x][1], ids[i])))
        label_dist.append({classes[i]: counter.get(i) for i in range(len(classes))})

    return ids, label_dist

def data_preprocessing(dataset: str, num_clients: int, num_iids: int, alpha: float, batch_size: int):
    trainset, testset = load_data(dataset)

    ids, dist = partition_data(trainset, num_clients=num_clients, num_iids=num_iids, alpha=alpha)
    trainloaders = []

    for i in range(num_clients):
        trainloaders.append(DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(ids[i])))

    testloaders = DataLoader(testset, batch_size=batch_size)
    return trainloaders, testloaders, ids, dist