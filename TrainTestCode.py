import torch
from torch import optim, cuda
from torchmetrics import Accuracy
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd
import os
import wandb
# Image manipulations

# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14

#custom class
from dataset import MyDataset
from slowfastnet import SlowFast,Bottleneck

# Printing out all outputs
InteractiveShell.ast_node_interactivity = 'all'
device='cuda'

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}') 
   
def getTop_k(test_loader,model):
    model.eval()
    top1 = []
    top5 = []
    accuracy1 = Accuracy(top_k=1)
    accuracy5 = Accuracy(top_k=5)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            # measure data loading time
#             print(f"Processing {batch_idx+1}/{len(test_loader)}")
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

            # compute output
            outputs = model(inputs)
            accuracy1(outputs, targets)
            accuracy5(outputs, targets)

    return accuracy1.compute(),accuracy5.compute() 

def accuracy(output, target,device, topk=(1, )):
    """Compute the topk accuracy(s)"""
    
    output = output.to(device)
    target = target.to(device)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Find the predicted classes and transpose
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []

        # For each k, find the percentage of correct
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def train(config,model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          scheduler,
          save_file_name,
          max_epochs_stop=10,
          n_epochs=20,
          print_every=2,device='cpu'):
    """Train a PyTorch Model

    Params
    --------
    Originally written by: https://github.com/WillKoehrsen/pytorch_challenge/blob/master/Transfer%20Learning%20in%20PyTorch.ipynb , but editted to fit our purpose
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats
        scheduler: Lr-scheduler to decay learning rate every 7 epochs
    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []



    overall_start = timer()
    
    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            data, target = data.to(device), target.to(device)
            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()
            scheduler.step(loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            
            # Need to convert correct tensor from int to float to average
            accuracy = (target == pred).sum()/target.shape[0]
            wandb.log({'batch_accuracy':accuracy.item()*100,'lr':current_lr})
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item()
            
            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                end='\r')
        
        # After training loops ends, start validation
        # else:
        

        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            # Validation loop
            for data, target in valid_loader:
                # Tensors to gpu
                
                data, target = data.to(device), target.to(device)

                # Forward pass
                output = model(data)

                # Validation loss
                loss = criterion(output, target)
                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)

                # Calculate validation accuracy
                _, pred = torch.max(output, dim=1)
                accuracy = (target == pred).sum()/target.shape[0]
                # Multiply average accuracy times the number of examples
                valid_acc += accuracy.item()

            # Calculate average losses
            train_loss = train_loss / len(train_loader)
            valid_loss = valid_loss / len(valid_loader)

            # Calculate average accuracy
            train_acc = train_acc / len(train_loader)
            valid_acc = valid_acc / len(valid_loader)

            history.append([train_loss, valid_loss, train_acc, valid_acc])
            #Logs
            wandb.log({'train_loss':train_loss,'train_accuracy':train_acc,'valid_acc':valid_acc,'valid_loss':valid_loss})
            # Print training and validation results
            if (epoch + 1) % print_every == 0:
                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                )

            # Save the model if validation loss decreases
            if valid_loss < valid_loss_min:
                # Save model
                torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                valid_best_acc = valid_acc
                best_epoch = epoch

            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                    )
                    total_time = timer() - overall_start
                    print(
                        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                    )

                    # Load the best state dict
                    model.load_state_dict(torch.load(save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer

                    # Format history
                    history = pd.DataFrame(
                        history,
                        columns=[
                            'train_loss', 'valid_loss', 'train_acc',
                            'valid_acc'
                        ])
                    return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history


