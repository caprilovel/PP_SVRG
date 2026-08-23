import os
import time
import torch 

import pandas as pds
import numpy as np
import wandb
import logging

from .utils import calculate_full_gradient, calculate_loss, update_weights, log_metrics, update_dataset
from .log import AverageCalculator, log_to_file
from .metric import accuracy




def train_one_epoch(model, optimizer, train_loader, train_loader_large, start_weights,
                    metric, loss_fn, model_snapshot=None, optimizer_snapshot=None,
                    temperature=0.5, optimize='SGD', device='cpu', update_weight=True, n_samples=None):
    
    
    logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if not update_weight:
        train_dataloader_copy =  update_dataset(3, train_loader, model, device, alpha=temperature, n_samples=n_samples)
        # train_dataloader_copy = train_loader   
    else:
        train_dataloader_copy = train_loader    
    
    if optimize == 'SVRG':
        g = calculate_full_gradient(model_snapshot, train_loader, start_weights, loss_fn, 
                                    optimizer_snapshot, device)
    elif optimize == 'SGD':
        g = calculate_full_gradient(model, train_loader, start_weights, loss_fn, optimizer,
                                    device)

    metric['grad'].update(g)
    
    if optimize == 'SVRG':
        u = optimizer_snapshot.get_param_groups()
        optimizer.set_u(u)
        
    
    weights = start_weights
    print("total iterations:", len(train_loader))
    logging.info("total iterations: %d", len(train_loader))
    for i, (images, labels) in enumerate(train_dataloader_copy):
        logging.info("Iteration: %d", i)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("Weights: ", weights)

        # Calculate loss and perform optimization
        loss_iter, yhat = calculate_loss(model, images, labels, weights, loss_fn, device)
        # if loss_iter.mean() > 100:
        #     print(yhat)   
            # print(weights)
        # if not update_weight:
        #     loss_iter += 1e-3 * sum(torch.norm(p, 2)**2 for p in model.parameters() if p.requires_grad and p.ndim > 1)
        optimizer.zero_grad()
        loss_iter.backward()    
        
        if optimize == 'SVRG':
            loss2, _ = calculate_loss(model_snapshot, images, labels, weights, loss_fn, device)
            optimizer_snapshot.zero_grad()         
            loss2.backward()
            optimizer.step(optimizer_snapshot.get_param_groups())
        else:
            optimizer.step()  
        
        if update_weight:
            with torch.no_grad():
                weights = update_weights(model, train_loader_large, loss_fn, 
                                 temperature, device)
        
        
        # Log metrics
        acc = accuracy(yhat.cpu(), labels)
        log_metrics(loss_iter, acc, metric, i)
    
    if optimize == 'SVRG':
        
        optimizer_snapshot.set_param_groups(optimizer.get_param_groups())
        
        
    return metric['loss'].avg, metric['acc'].avg, metric['grad'].avg, weights

def train_model(model, model_snapshot, optimizer, optimizer_snapshot, train_loader,
                train_loader_large, val_loader, loss_fn, log_dir, n_epochs, optimize,
                print_interval, temperature, device, log, use_wandb, update_weight, n_samples=None,
                warmup_epochs=0, warmup_lr=None, *args, **kwargs):

    start_weights = update_weights(model, train_loader_large, loss_fn, beta=temperature, device=device)
    metrics = {
        'loss': AverageCalculator(),
        'acc': AverageCalculator(),
        'grad': AverageCalculator(),
    }

    columns = ['epoch', 'phase', 'train_loss', 'train_acc', 'eval_loss', 'eval_acc', 'weights', 'grads']
    df = pds.DataFrame(columns=columns)

    global_epoch = 0

    if optimize == 'SVRG' and warmup_epochs > 0:
        from optim.sgd import SGD_Simple

        base_group = optimizer.param_groups[0]
        lr = warmup_lr if warmup_lr is not None else base_group['lr']
        warmup_optimizer = SGD_Simple(model.parameters(), lr=lr, weight_decay=base_group['weight_decay'])

        print(f"Warming up with SGD for {warmup_epochs} epoch(s) before starting SVRG (warmup_lr={lr})")

        for w_epoch in range(warmup_epochs):
            t0 = time.time()

            train_loss, train_acc, grads, new_weights = train_one_epoch(
                    model, warmup_optimizer, train_loader, train_loader_large, start_weights, metrics,
                    loss_fn, model_snapshot=None, optimizer_snapshot=None, temperature=temperature,
                    optimize='SGD', device=device, n_samples=n_samples
                )
            eval_loss, eval_acc = eval_one_epoch(model, val_loader, loss_fn, device, temperature)

            if use_wandb:
                wandb.log({
                    'epoch': global_epoch,
                    'phase': 'warmup',
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'eval_loss': eval_loss,
                    'eval_acc': eval_acc,
                    'grads': grads,
                    'weights': new_weights
                })

            new_row = {
                'epoch': global_epoch,
                'phase': 'warmup',
                'train_loss': train_loss,
                'train_acc': train_acc,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc,
                'weights': new_weights,
                'grads': grads
            }
            df = pds.concat([df, pds.DataFrame(new_row, index=[0])], ignore_index=True)
            print(df)

            if w_epoch % print_interval == 0:
                print(f"[Warmup] Epoch {w_epoch} / {warmup_epochs}, train loss: {train_loss}, train acc: {train_acc}, grads: {grads}, time: {time.time() - t0}")

            start_weights = new_weights
            global_epoch += 1

            if log:
                df.to_csv(os.path.join(log_dir, 'train_stats.csv'))

        # sync the SVRG snapshot model with the warmed-up model before starting SVRG
        model_snapshot.load_state_dict(model.state_dict())

    for epoch in range(n_epochs):
        t0 = time.time()


        train_loss, train_acc, grads, new_weights = train_one_epoch(
                model, optimizer, train_loader, train_loader_large, start_weights, metrics,
                loss_fn, model_snapshot, optimizer_snapshot, temperature,
                optimize=optimize, device=device, n_samples=n_samples
            )
        eval_loss, eval_acc = eval_one_epoch(model, val_loader, loss_fn, device, temperature)

        if use_wandb:
            wandb.log({
                'epoch': global_epoch,
                'phase': optimize,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc,
                'grads': grads,
                'weights': new_weights
            })

        new_row = {
            'epoch': global_epoch,
            'phase': optimize,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
            'weights': new_weights,
            'grads': grads
        }
        # use concat
        df = pds.concat([df, pds.DataFrame(new_row, index=[0])], ignore_index=True)
        print(df)

        if epoch % print_interval == 0:
            print(f"Epoch {epoch} / {n_epochs}, train loss: {train_loss}, train acc: {train_acc}, grads: {grads}, new weights: {new_weights}, time: {time.time() - t0}")



        start_weights = new_weights
        global_epoch += 1

        if (epoch + 1) % 1 == 0 and log:
            df.to_csv(os.path.join(log_dir, 'train_stats.csv'))
    if log:
        open(os.path.join(log_dir, 'done'), 'a').close()

def eval_one_epoch_fixed(model, val_loader, loss_fn, device):
    """Evaluate on fixed original distribution (no performative shift)."""
    import copy
    val_loader_copy = copy.deepcopy(val_loader)
    metrics = {
        'loss': AverageCalculator(),
        'acc': AverageCalculator(),
    }
    with torch.no_grad():
        for images, labels in val_loader_copy:
            images, labels = images.to(device), labels.to(device)
            loss_iter, yhat = calculate_loss(model, images, labels, None, loss_fn, device)
            acc = accuracy(yhat.cpu(), labels.cpu())
            log_metrics(loss_iter, acc, metrics, None)
    return metrics['loss'].avg, metrics['acc'].avg


def eval_one_epoch(model, val_loader, loss_fn, device, temperature=0.5, n_samples=None):
    val_loader_copy = update_dataset(3, val_loader, model, device, alpha=temperature, n_samples=n_samples)
    metrics = {
        'loss': AverageCalculator(),
        'acc': AverageCalculator(),
    }
    with torch.no_grad():
        for images, labels in val_loader_copy:
            loss_iter, yhat = calculate_loss(model, images, labels, None, loss_fn, device)
            acc = accuracy(yhat.cpu(), labels.cpu())
            log_metrics(loss_iter, acc, metrics, None)
    return metrics['loss'].avg, metrics['acc'].avg
        
def train_credit_model(model, model_snapshot, optimizer, optimizer_snapshot, train_loader,
                train_loader_large, val_loader, loss_fn, log_dir, n_epochs, optimize,
                print_interval, temperature, device, log, use_wandb, n_samples=None):

    # update_dataset(3, train_loader, model, device, alpha=temperature)

    metrics = {
        'loss': AverageCalculator(),
        'acc': AverageCalculator(),
        'grad': AverageCalculator(),
    }

    columns = ['epoch', 'train_loss', 'train_acc', 'eval_loss', 'eval_acc', 'weights', 'grads']
    df = pds.DataFrame(columns=columns)

    for epoch in range(n_epochs):
        t0 = time.time()
        train_loss, train_acc, grads, new_weights = train_one_epoch(
                model, optimizer, train_loader, train_loader_large, None, metrics,
                loss_fn, model_snapshot, optimizer_snapshot, temperature,
                optimize=optimize, device=device, update_weight=False, n_samples=n_samples
            )
        eval_loss, eval_acc = eval_one_epoch(model, val_loader, loss_fn, device, temperature, n_samples=n_samples)

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc,
                'grads': grads,
                'weights': new_weights
            })

        # for metric in metrics.values():
        #     metric.reset()

        new_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
            'weights': new_weights,
            'grads': grads
        }
        # use concat
        df = pds.concat([df, pds.DataFrame(new_row, index=[0])], ignore_index=True)
        print(df)

        if epoch % print_interval == 0:
            print(f"Epoch {epoch} / {n_epochs}, train loss: {train_loss}, train acc: {train_acc}, eval_loss: {eval_loss}, eval_acc: {eval_acc}, grads: {grads}, new weights: {new_weights}, time: {time.time() - t0}")
            
            


        if (epoch + 1) % 1 == 0 and log:
            df.to_csv(os.path.join(log_dir, 'train_stats.csv'))
    if log:
        open(os.path.join(log_dir, 'done'), 'a').close()