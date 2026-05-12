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
    model_k.eval()
    group_loss = {}
    group_weights = {}
    
    # logging.basicConfig(
    # format='%(asctime)s - %(levelname)s - %(message)s',
    # level=logging.INFO,
    # datefmt='%Y-%m-%d %H:%M:%S'
    # )
    # logging.info("start generate sub dataset")
    
    # sub_dataloader = get_sub_dataloader(train_loader_large, s_ratio, device)
    
    # logging.info("start generate group loss")
    print("dataloader size: ", len(train_loader_large), len(train_loader_large.dataset))
    # print("sub_dataloader size: ", len(sub_dataloader), len(sub_dataloader.dataset))
    
    for images, labels in train_loader_large:
    #     pass
    # logging.info("median group loss")
    # for images, labels in sub_dataloader:
        images = images.to(device)
        yhat = model_k(images)
        labels = labels.to(device)
        # loss for each sample without averaging
        loss_iter = loss_fn(reduction='none')(yhat, labels)
        loss_iter = loss_iter.view(-1)  # Ensure loss_iter is a 1D tensor
        unique_labels = torch.unique(labels)
        for i in range(len(unique_labels)):
            label = unique_labels[i].item()
            if label not in group_loss:
                group_loss[label] = 0.0
                group_weights[label] = 0
            
            group_loss[label] += loss_iter[labels == label].mean()
            group_weights[label] += (labels == label).sum().item()
            
    for label in group_loss:
        group_loss[label] /= group_weights[label]        
    # logging.info("end generate group loss ")
    # print("Group losses: ", group_loss)
    max_loss = max(group_loss.values())
    # calculate softmax of -beta * loss
    group_weights = {
        label: np.exp(-beta * (group_loss[label] - max_loss).item()) for label in group_loss
        }
    total = sum(group_weights.values())
    
    
    
    for label in group_weights:
        group_weights[label] /= total
        group_weights[label] *= len(group_weights)  # scale to the number of groups
    # print("Group weights: ", group_weights)
    
    return group_weights

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
    if weights is not None:
        label_weights = torch.tensor([weights[label] for label in weights.keys()], dtype=torch.float32).to(device)
    else:
        label_weights = None
    yhat = model(images)
    loss_iter = loss_fn(weight=label_weights)(yhat, labels)
    # loss_iter = loss_fn()(yhat, labels)
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
     