import sys
import os
import csv
import argparse
import torch
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dloader
from Logger import Logger
from InferenceDataset import InferenceDataset

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('--run_name', type=str, required=False, default='customdataset_inference_testing',
                    help='name that will be used for this instance of the inference run')
parser.add_argument('--dataset_directory', type=str, required=False, default='data/inference_data',
                    help='path to the inference dataset directory')
parser.add_argument('--model_directory', type=str, required=False, default='runs/customdataset_training_testing',
                    help='path to the inference models folder')
args = parser.parse_args()

# global parameters
run_name = "runs/" + args.run_name
data_folder = args.dataset_directory
model_dir = args.model_directory
model_path = model_dir + "/model.pt"

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

# load inference images and create dataloader
batch_size = 8
inference_data = InferenceDataset(data_folder, transform=transform)
dataloader = dloader.DataLoader(inference_data, batch_size=batch_size, shuffle=False)

# load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.eval()
model.to(device)

# load class ordering
with open(model_dir + '/class_order.csv', 'r') as f:
    reader = csv.reader(f)
    class_order = list(reader)[0]

# get predicted labels from data
predictions = []
for imgs in dataloader:
    imgs = imgs.to(device)

    output = model(imgs)
    _, preds = torch.max(output, 1)

    # store model predictions as class labels
    for p in preds:
        predictions.append(class_order[p.item()])

# save predictions in csv file -- file1,classA /n file2,classC /n ...ect
pred_pairs = list(zip(inference_data.file_names, predictions))
with open(run_name + '/predictions.csv', 'w') as f:
    wr = csv.writer(f)
    wr.writerows(pred_pairs)

print("=========================")
print("Predictions saved to:")
print(run_name + '/predictions.csv')
print("=========================")
