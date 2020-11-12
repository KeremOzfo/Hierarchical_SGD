import torch
import math
import time
import numpy as np


def pull_model(model_user, model_server):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_user.data = param_server.data[:] + 0
    return None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def zero_grad_ps(model):
    for param in model.parameters():
        param.grad = torch.zeros_like(param.data)

    return None


def push_grad(model_user, model_server, num_cl):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_server.grad.data += param_user.grad.data / num_cl
    return None


def push_model(model_user, model_server, num_cl):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_server.data += param_user.data / num_cl
    return None

def push_models(model_users, model_server, num_cl):
    for model_user in model_users:
        for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
            param_server.data += param_user.data / num_cl
    return None



def initialize_zero(model):
    for param in model.parameters():
        param.data.mul_(0)
    return None


def update_model(model, prev_model, lr, momentum, weight_decay):
    for param, prevIncrement in zip(model.parameters(), prev_model.parameters()):
        incrementVal = param.grad.data.add(weight_decay, param.data)
        incrementVal.add_(momentum, prevIncrement.data)
        incrementVal.mul_(lr)
        param.data.add_(-1, incrementVal)
        prevIncrement.data = incrementVal
    return None


def get_grad_flattened(model, device):
    grad_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        if p.requires_grad:
            a = p.grad.data.flatten().to(device)
            grad_flattened = torch.cat((grad_flattened, a), 0)
    return grad_flattened


def get_model_flattened(model, device):
    model_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        a = p.data.flatten().to(device)
        model_flattened = torch.cat((model_flattened, a), 0)
    return model_flattened


def get_model_sizes(model):
    # get the size of the layers and number of eleents in each layer.
    # only layers that are trainable
    net_sizes = []
    net_nelements = []
    for p in model.parameters():
        if p.requires_grad:
            net_sizes.append(p.data.size())
            net_nelements.append(p.nelement())
    return net_sizes, net_nelements


def unshuffle(shuffled_vec, seed):
    orj_vec = torch.empty(shuffled_vec.size())
    perm_inds = torch.tensor([i for i in range(shuffled_vec.nelement())])
    perm_inds_shuffled = shuffle_deterministic(perm_inds, seed)
    for i in range(shuffled_vec.nelement()):
        orj_vec[perm_inds_shuffled[i]] = shuffled_vec[i]
    return orj_vec


def shuffle_deterministic(grad_flat, seed):
    # Shuffle the list ls using the seed `seed`
    torch.manual_seed(seed)
    idx = torch.randperm(grad_flat.nelement())
    return grad_flat.view(-1)[idx].view(grad_flat.size())


def get_indices(net_sizes, net_nelements):
    # for reconstructing grad from flattened grad
    ind_pairs = []
    ind_start = 0
    ind_end = 0
    for i in range(len(net_sizes)):

        for j in range(i + 1):
            ind_end += net_nelements[j]
        # print(ind_start, ind_end)
        ind_pairs.append((ind_start, ind_end))
        ind_start = ind_end + 0
        ind_end = 0
    return ind_pairs


def make_grad_unflattened(model, grad_flattened, net_sizes, ind_pairs):
    # unflattens the grad_flattened into the model.grad
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            temp = grad_flattened[ind_pairs[i][0]:ind_pairs[i][1]]
            p.grad.data = temp.reshape(net_sizes[i])
            i += 1
    return None


def make_model_unflattened(model, model_flattened, net_sizes, ind_pairs):
    # unflattens the grad_flattened into the model.grad
    i = 0
    for p in model.parameters():
        temp = model_flattened[ind_pairs[i][0]:ind_pairs[i][1]]
        p.data = temp.reshape(net_sizes[i])
        i += 1
    return None


def make_sparse_grad(grad_flat, sparsity_window, device):
    # sparsify using block model
    num_window = math.ceil(grad_flat.nelement() / sparsity_window)

    for i in range(num_window):
        ind_start = i * sparsity_window
        ind_end = min((i + 1) * sparsity_window, grad_flat.nelement())
        a = grad_flat[ind_start: ind_end]
        ind = torch.topk(a.abs(), k=1, dim=0)[1]  # return index of top not value
        val = a[ind]
        ind_true = ind_start + ind
        grad_flat[ind_start: ind_end] *= torch.zeros(a.nelement()).to(device)
        grad_flat[ind_true] = val

    return None


def adjust_learning_rate(optimizer, epoch, lr_change, lr):
    lr_change = np.asarray(lr_change)
    loc = np.where(lr_change == epoch)[0][0] + 1
    lr *= (0.1 ** loc)
    lr = round(lr, 3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_LR(optimizer):
    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr


def lr_warm_up(optimizers, num_workers, epoch, start_lr):
    for cl in range(num_workers):
        for param_group in optimizers[cl].param_groups:
            if epoch == 0:
                param_group['lr'] = 0.1
            else:
                lr_change = (start_lr - 0.1) / 4
                param_group['lr'] = (lr_change * epoch) + 0.1

