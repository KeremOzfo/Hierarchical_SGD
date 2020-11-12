import torch
from torch.utils.data import DataLoader
# custom modules
import data_loader as dl
from nn_classes import *
import server_functions as sf
import math
from parameters import *
import time
import numpy as np
from tqdm import tqdm
from torchsummary import summary


def evaluate_accuracy(model, testloader, device):
    """Calculates the accuracy of the model"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def train(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    #summary(net_ps,(3,32,32))
    net_ps_prev = get_net(args).to(device)
    sf.initialize_zero(net_ps_prev)
    prev_models = [get_net(args).to(device) for u in range(num_client)]
    [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]
    net_clusters = [get_net(args).to(device) for u in range(args.num_cluster)]
    M_clusters = [get_net(args).to(device) for u in range(args.num_cluster)]
    [sf.initialize_zero(M_vals) for M_vals in M_clusters]

    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4) for cl in
                  range(num_client)]
    schedulers = [torch.optim.lr_scheduler.MultiStepLR(optimizers[cl], milestones=args.lr_change, gamma=0.1) for cl in range(num_client)]

    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)
    trainLoader_all = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    accuracys = []
    runs = math.ceil(N_s / (args.bs * num_client * args.LocalIter))
    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    assert N_s/num_client > args.LocalIter * args.bs
    assert num_client % args.num_cluster == 0
    worker_per_cluster = int(num_client/args.num_cluster)
    for epoch in tqdm(range(args.num_epoch)):
        atWarmup = (args.warmUp and epoch < 5)
        if atWarmup:
            sf.lr_warm_up(optimizers, args.num_client, epoch, args.lr)

        for run in range(runs):
            for cl in range(num_client):
                localIter = 0

                trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                         shuffle=True)
                for data in trainloader:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizers[cl].zero_grad()
                    predicts = net_users[cl](inputs)
                    loss = criterions[cl](predicts, labels)
                    loss.backward()
                    optimizers[cl].step()
                    localIter +=1
                    if localIter == args.LocalIter:
                        break
            sf.initialize_zero(net_ps)
            lr = sf.get_LR(optimizers[0])
            for i,cluster_model in enumerate(net_clusters):
                start = i * worker_per_cluster
                if args.SlowMo:
                    flattened_cluster = sf.get_model_flattened(cluster_model, device)
                    flattened_momentum = sf.get_model_flattened(M_clusters[i],device)
                    avg_model = torch.zeros_like(flattened_cluster).to(device)
                    for model in net_users[start:start+worker_per_cluster]:
                        avg_model.add_(1/worker_per_cluster
                                       ,sf.get_model_flattened(model,device))
                    psuedo_grad = flattened_cluster.sub(1,avg_model)
                    psuedo_grad.mul_(1/lr)
                    psuedo_grad.add_(args.beta,flattened_momentum)
                    sf.make_model_unflattened(M_clusters[i],psuedo_grad,net_sizes,ind_pairs)
                    flattened_cluster.sub_(args.alfa * lr, psuedo_grad)
                    sf.make_model_unflattened(cluster_model,flattened_cluster,net_sizes,ind_pairs)

                else:
                    sf.initialize_zero(cluster_model)
                    sf.push_models(net_users[start: start+worker_per_cluster],
                                   cluster_model,worker_per_cluster)
                sf.push_model(cluster_model,net_ps,args.num_cluster)
            [sf.pull_model(client,net_ps) for client in net_users]


        acc = evaluate_accuracy(net_ps, testloader, device)
        accuracys.append(acc * 100)
        print('accuracy:{}'.format(acc*100))
        if not atWarmup:
            [schedulers[cl].step() for cl in range(num_client)] ## adjust Learning rate
    return accuracys



