import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, datasets 
import numpy as np
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
import logging

def groupwise_weights(
    model_k, train_loader_large, loss_fn, beta=1, device='cpu', s_ratio=1
    # model_k, train_loader_large, loss_fn, beta=0.5, device='cpu'
    ):
    """
    first calculate the loss of each label group wrt model_k
    then calculate each group's weight as softmax of -beta * loss
    also calculate the average gradient norm
    return a dictionary with keys = label, values = weights
    """
    if beta == 0:
        # softmax(-beta * loss) is uniform regardless of the per-group losses,
        # so skip the full-dataset forward pass entirely (this is the hot path
        # called once per training iteration and dominates runtime at large n).
        all_labels = set()
        for _, labels in train_loader_large:
            all_labels.update(labels.tolist())
        return {label: 1.0 for label in sorted(all_labels)}

    model_k.eval()

    print("dataloader size: ", len(train_loader_large), len(train_loader_large.dataset))

    # accumulate per-class loss sum / count as GPU tensors (indexed by class id)
    # -- vectorized, no per-class .item() calls in the hot loop, since each
    # .item()/.cpu() forces a host<->device sync and this function is called
    # once per training iteration.
    loss_sum = None
    loss_count = None
    for images, labels in train_loader_large:
        images = images.to(device)
        labels = labels.to(device)
        yhat = model_k(images)
        loss_iter = loss_fn(reduction='none')(yhat, labels).view(-1)

        if loss_sum is None:
            num_classes = int(labels.max().item()) + 1
            loss_sum = torch.zeros(num_classes, device=device)
            loss_count = torch.zeros(num_classes, device=device)

        loss_sum.scatter_add_(0, labels, loss_iter)
        loss_count.scatter_add_(0, labels, torch.ones_like(loss_iter))

    present = loss_count > 0
    group_loss = loss_sum[present] / loss_count[present]
    labels_present = present.nonzero(as_tuple=True)[0]

    max_loss = group_loss.max()
    group_weights_t = torch.exp(-beta * (group_loss - max_loss))
    group_weights_t = group_weights_t / group_weights_t.sum() * group_weights_t.numel()

    # single host<->device sync at the end, for the (num_classes,)-sized result
    labels_cpu = labels_present.tolist()
    weights_cpu = group_weights_t.tolist()
    return dict(zip(labels_cpu, weights_cpu))

def get_sub_dataloader(dataloader, ratio, device, use_full=False):
    all_data_by_class = defaultdict(list)
    batch_size = dataloader.batch_size
    
    for images, labels in dataloader:
        for img, label in zip(images, labels):
            all_data_by_class[label.item()].append(img)

    min_class_size = min(len(v) for v in all_data_by_class.values())
    num_per_class = max(1, int(min_class_size * ratio))

    sampled_images = []
    sampled_labels = []

    for label, images in all_data_by_class.items():
        indices = torch.randperm(len(images))[:num_per_class]
        sampled_images.extend([images[i] for i in indices])
        sampled_labels.extend([label] * num_per_class)

    sampled_images = torch.stack(sampled_images).to(device)
    sampled_labels = torch.tensor(sampled_labels).to(device)
    print('nums of labels:', len(sampled_labels))

    sampled_dataset = TensorDataset(sampled_images, sampled_labels)
    if use_full:
        sub_loader = DataLoader(sampled_dataset, batch_size=len(sampled_dataset), shuffle=False)
    else:
        sub_loader = DataLoader(sampled_dataset, batch_size=batch_size, shuffle=True)
    return sub_loader

def calculate_loss(model, images, labels, weights, loss_fn, device):
    images = images.to(device)
    labels = labels.to(device)
    if weights is not None and len(set(weights.values())) > 1:
        # skip building+transferring a weight tensor when every class already
        # has the same weight (e.g. temperature=0) -- weight=None is exactly
        # equivalent and avoids a per-call CPU->GPU sync that otherwise
        # dominates runtime at small batch sizes (this function is called
        # once per iteration, plus once per batch in calculate_full_gradient).
        label_weights = torch.tensor([weights[label] for label in weights.keys()], dtype=torch.float32).to(device)
    else:
        label_weights = None
    yhat = model(images)
    loss_iter = loss_fn(weight=label_weights)(yhat, labels)
    return loss_iter, yhat

def update_weights(model, train_loader_large, loss_fn, beta, device, s_ratio=1):
    return groupwise_weights(model, train_loader_large, loss_fn, beta=beta, device=device, s_ratio=s_ratio)

def log_metrics(loss_iter, acc, metric, iter):
    metric['loss'].update(loss_iter.data.item())
    metric['acc'].update(acc)
    print(f"iteration{iter}:loss: {loss_iter.data.item()}, acc: {acc}")

def calculate_full_gradient(model, train_loader, start_weights, loss_fn, optimizer, device):
    model.train()
    optimizer.zero_grad()
    for images, labels in train_loader:
        loss_iter, _ = calculate_loss(model, images, labels, start_weights, loss_fn, device)
        loss_iter.backward()
    
    for param in model.parameters():
        if param.grad is not None:
            param.grad /= len(train_loader)
    
    full_grd = torch.cat([param.grad.view(-1) for param in model.parameters()])
    g = ((full_grd.norm(2))**2).item()
    print("full gradient norm: ", g)
    return g

# def update_dataset(columns, dataloader, model, device, alpha=0.1):
#     model.eval()

#     dataset = dataloader.dataset  
    
#     all_data = dataset.data.to(device)  
#     all_labels = dataset.labels.to(device)

#     for idx in range(0, len(dataset), dataloader.batch_size):
#         end_idx = min(idx + dataloader.batch_size, len(dataset)) 
#         i_data = all_data[idx:end_idx]
#         labels = all_labels[idx:end_idx]

#         i_data = i_data.clone().detach().requires_grad_(True)

#         outputs = model(i_data)
#         loss = F.cross_entropy(outputs, labels)
#         loss.backward()

#         grads = i_data.grad
#         i_data_updated = i_data.clone().detach()
#         i_data_updated[:, :columns] += alpha * grads[:, :columns]
#         dataset.data[idx:end_idx] = i_data_updated.detach().cpu()


        
# def _update_dataset(columns, dataloader, model, device, alpha=0.1):
#     model.eval()


#     dataset = dataloader_copy.dataset  
#     full_dataset = dataset.dataset  
#     indices = dataset.indices      

#     # assuming full_dataset.data and full_dataset.labels are Tensor
#     all_data = full_dataset.data[indices].to(device)
#     all_labels = full_dataset.labels[indices].to(device)

#     batch_size = dataloader.batch_size
#     for i, idx in enumerate(range(0, len(indices), batch_size)):
#         end_idx = min(idx + batch_size, len(indices))
#         i_data = all_data[i * batch_size : (i + 1) * batch_size]
#         labels = all_labels[i * batch_size : (i + 1) * batch_size]

#         i_data = i_data.clone().detach().requires_grad_(True)

#         outputs = model(i_data)
#         loss = F.cross_entropy(outputs, labels)
#         loss.backward()

#         grads = i_data.grad
#         i_data_updated = i_data.clone().detach()
#         i_data_updated[:, :columns] += alpha * grads[:, :columns]

#         # restore the updated data back to the original dataset
#         full_dataset.data[indices[i * batch_size : (i + 1) * batch_size]] = i_data_updated.detach().cpu()


def update_dataset(columns, dataloader, model, device, alpha=0.1, n_samples=None):
    from torch.utils.data import Subset, TensorDataset
    model.eval()

    dataloader_copy = copy.deepcopy(dataloader)
    dataset = dataloader_copy.dataset

    if isinstance(dataset, Subset):
        full_dataset = dataset.dataset
        indices = dataset.indices
        all_data = full_dataset.data[indices].to(device)
        all_labels = full_dataset.labels[indices].to(device)
    elif isinstance(dataset, TensorDataset):
        full_dataset = dataset
        indices = list(range(len(dataset)))
        all_data = dataset.tensors[0].to(device)
        all_labels = dataset.tensors[1].to(device)
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")

    if n_samples is not None and n_samples < len(indices):
        selected = torch.randperm(len(indices))[:n_samples].tolist()
    else:
        selected = list(range(len(indices)))

    batch_size = dataloader.batch_size
    for i, idx in enumerate(range(0, len(selected), batch_size)):
        batch_selected = selected[idx : idx + batch_size]
        i_data = all_data[batch_selected]
        labels = all_labels[batch_selected]

        i_data = i_data.clone().detach().requires_grad_(True)

        outputs = model(i_data)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        grads = i_data.grad
        # normalize the gradients in the sample level
        # grads = grads / (grads.norm(dim=1, keepdim=True) + 1e-8)

        i_data_updated = i_data.clone().detach()
        i_data_updated[:, :columns] += alpha * grads[:, :columns]

        if isinstance(dataset, Subset):
            orig_indices = [indices[j] for j in batch_selected]
            full_dataset.data[orig_indices] = i_data_updated.detach().cpu()
        else:
            full_dataset.tensors[0][batch_selected] = i_data_updated.detach()
    return dataloader_copy
     