import seaborn as sns
# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch.autograd import Variable
import pandas as pd
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from natsort import natsorted

# Data science tools
import numpy as np
import os

# Image manipulations
from PIL import Image
# Useful for examining network
from torchsummary import summary
# Timing utility
from timeit import default_timer as timer

# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

# Location of data
fileDir = os.path.dirname(os.path.realpath('__file__'))
print(fileDir)
datadir = fileDir
traindir = datadir + '/train/'
validdir = datadir + '/validate/'
testdir = datadir + '//test//'

save_file_name = 'vgg16-transfer-4.pt'
checkpoint_path = 'vgg16-transfer-4.pth'

# Change to fit hardware
batch_size = 50

# Whether to train on a gpu
train_on_gpu = cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

# Number of gpus
if train_on_gpu:
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')
    if gpu_count > 1:
        multi_gpu = True
    else:
        multi_gpu = False

# map_location=torch.device('cpu')

# Empty lists
categories = []
img_categories = []
n_train = []
n_valid = []
# n_test = []
hs = []
ws = []

def load_checkpoint(path):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    # Get the model name
    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    # Load in checkpoint
    checkpoint = torch.load(path)

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')

    # Move to gpu
    if multi_gpu:
        model = nn.DataParallel(model)

    if train_on_gpu:
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

model, optimizer = load_checkpoint(path=checkpoint_path)





# Image transformations
image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# input = transform(image_names)
# data = {
#     'test':
#     datasets.ImageFolder(root=testdir, transform=image_transforms['test'])}

# unsqueeze batch dimension, in case you are dealing with a single image
# input = input.unsquueeze(0)

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

my_dataset = CustomDataSet(testdir, transform=image_transforms['test'])
train_loader = torch.utils.data.DataLoader(my_dataset , batch_size=batch_size, shuffle=False,
                               num_workers=4, drop_last=True)


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = image_transforms['test'](image)
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

model.eval()

result_df = pd.DataFrame(columns=['img_name', 'label'])
sm = torch.nn.Softmax()
max = 0
for filename in os.listdir(testdir):
    # if max == 50:
    #     break
    max +=1
    image = image_loader(f'{testdir}/{filename}')
    label = torch.argmin((model(image))).item()
    label = int(label)
    label +=1
    # print(label)
    result_df = result_df.append({'img_name': filename, 'label': label}, ignore_index=True)

# print(len(image_names))
inbetween_df = pd.DataFrame(result_df.img_name.str.split('_',1).tolist(),
                                 columns = ['x','y'])
inbetween_df2 = pd.DataFrame(inbetween_df.y.str.split('.',1).tolist(),
                                 columns = ['number','type'])
result_df = result_df.join(inbetween_df2)
result_df['number'] = result_df['number'].astype(int)
result_df.sort_values('number', inplace=True)
print(result_df.head())

result_df[['img_name', 'label']].to_csv('output.csv')




