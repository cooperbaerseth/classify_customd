import copy
import sys
import os
import csv
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data.dataloader as dloader
import torchvision.models as models
import torchvision.transforms as transforms
from torchsummary import summary

from CustomDataset import CustomDataset
from Logger import Logger

# This function will take a pretrained model, replace the fully connected layers at the end, and train them with new
#   data from scratch.
def transfer_train(model, train, valid, num_epochs=25, quick=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # n_classes = len(np.unique(train.target))
    n_classes = len(train.dataset.class_list)
    fc_feats = 1664 # for densenet
    dataset_sizes = {'train': len(train), 'valid': len(valid)}

    # Replace classifier of model with one that corresponds to the new dataset
    transfer_classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=fc_feats, out_features=n_classes, bias=True)
    )
    model.classifier = transfer_classifier
    model.to(device)
    summary(model, (3,224,224))


    # TRAIN THE NEW CLASSIFIER
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimize only classifier parameters
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # When training, it helps to do some data augmentation, so add random flip/rotation to train transform
    train_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(10),
                                          transforms.ToTensor(),
                                          transforms.Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225]
                                          )])
    train.transform = train_transform

    # Create dataloaders
    train_dataloader = dloader.DataLoader(train, batch_size=8, shuffle=False)
    valid_dataloader = dloader.DataLoader(valid, batch_size=8, shuffle=True)
    dataloaders = {'train': train_dataloader,
                   'valid': valid_dataloader}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    converge_count = 0
    converge_thresh = 4
    converge_accu = 0.0
    for epoch in range(num_epochs):
        print("-" * 10)
        print("Epoch " + str(epoch) + "/" + str(num_epochs-1))
        print("-" * 10)

        train_accu = 0.0
        valid_accu = 0.0

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for iter, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                if iter % 500 == 0:
                    print(str(iter) + '/' + str(dataset_sizes[phase]) + ' iterations')

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(str(phase) + " Loss: " + str(round(epoch_loss, 4)) + " Acc: " + str(round(epoch_acc.item(), 4)) + '\n')

            if phase == 'train':
                train_accu = epoch_acc.item()
            else:
                valid_accu = epoch_acc.item()

                # if the accuracy hasn't changed significantly in the last 4 epochs, consider the model converged
                if quick:
                    if round(best_acc, 2) == round(valid_accu, 2):
                        converge_count += 1
                        if converge_count > converge_thresh:
                            model.load_state_dict(best_model_wts)
                            return model, best_acc

            # deep copy the model
            if phase == 'valid' and epoch_acc.item() > best_acc:
                best_acc = epoch_acc.item()
                best_model_wts = copy.deepcopy(model.state_dict())

            # if model is performing above 95% accurate, end training early
            if quick:
                if train_accu > 0.95 and valid_accu > 0.95:
                    model.load_state_dict(best_model_wts)
                    return model, best_acc

    model.load_state_dict(best_model_wts)
    return model, best_acc

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, required=False, default='customdataset_training_testing',
                    help='name that will be used for this instance of the training run')
parser.add_argument('--dataset_directory', type=str, required=False, default='data/training_data',
                    help='path to the training dataset directory')
parser.add_argument('--num_trials', type=int, required=False, default=10,
                    help='number of times the neural net will be trained... returns best from these')
args = parser.parse_args()

# global parameters
run_name = 'runs/' + args.run_name
dataset_root = args.dataset_directory
num_trials = args.num_trials

# create run directory
if not os.path.exists(run_name):
    os.makedirs(run_name)

# send console prints to file
sys.stdout = Logger(run_name)

print("************* Start " + run_name + " *************")

# images have to be transformed in a certain way for the pretrained model
transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(224),
 transforms.ToTensor(),
 transforms.Normalize(
 mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225]
 )])

# surveySite dataset
custom_dataset = CustomDataset(root=dataset_root, transform=transform)
custom_dataset.split_dataset(per_class='min')
train_size = round(0.7*len(custom_dataset))

# save class order for inference later
with open(run_name + '/class_order.csv', 'w') as class_order:
    wr = csv.writer(class_order)
    wr.writerow(custom_dataset.class_list)

accuracies = np.zeros(num_trials)
best_accu = 0.0
for t in range(num_trials):
    print("=========================")
    print("TRIAL: " + str(t) + '/' + str(num_trials))
    print("=========================")

    # resample dataset for each trial
    trainset, validset = torch.utils.data.random_split(custom_dataset, [train_size, len(custom_dataset) - train_size])

    dataloader = dloader.DataLoader(validset, batch_size=5, shuffle=False)

    # import the model and set it to evaluation mode (so no weights are changed)
    model = models.densenet169(pretrained=True)
    model.eval()

    model_trained, accu = transfer_train(model, trainset, validset)
    accuracies[t] = accu

    if accu > best_accu:
        best_model = model_trained
        best_accu = accu

print("=========================")
print(":::ACCURACY RESULTS:::")
print("Mean: " + str(np.mean(accuracies)))
print("Std: " + str(np.std(accuracies)))
print("Best Accu: " + str(best_accu))
print("=========================")

# save config file that shows details on the current network
config = open(run_name + '/config.txt', 'w')
config.write("===========================\n")
config.write("Trained Model Configuration\n")
config.write("===========================\n")
config.write("Model Name: " + run_name + '\n')
config.write("Date/Time Created: " + str(datetime.datetime.now()) + '\n')
config.write("Accuracy: " + str(best_accu) + '\n')
config.write("Classes Trained On: \n")
for i, c in enumerate(custom_dataset.class_list):
    config.write(str(i) + ') ' + c)
    config.write('\n')
config.close()

# save model
torch.save(best_model, run_name + "/model.pt")

