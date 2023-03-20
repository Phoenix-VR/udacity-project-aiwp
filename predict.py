import argparse
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision.transforms.functional as fn
import json
from torch import nn

# The file should be run by command in following format:----
# $ python predict.py image_path checkpoint_file (--gpu and other optional arguments)
# $ python predict.py flowers/train/101.jpg checkpointsmodel.pth

# Predict flower name from an image with predict.py along with the probability of that name. 
# That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

# GETTING THE COMMAND LINE ARGUMENTS INPUT

parser = argparse.ArgumentParser()
parser.add_argument("image_path",type=str,default="flowers/train/101.jpg")
parser.add_argument("checkpoint",default="checkpoint.pth")
parser.add_argument("--top_k",default="3",type=int)
parser.add_argument("--category_names",default="cat_to_name.json")
parser.add_argument("--gpu",action='store_true')

args = parser.parse_args()
device = "cuda" if args.gpu else "cpu"

# LOADING OUR MODEL 

checkpoint = torch.load(args.checkpoint)
model_arch = checkpoint["model_arch"]

# BUILDING OUR MODEL ARCHITECHTURE JUST LIKE WE DID WHILE TRAINING (IN train.py)

if model_arch == "resnet":
    model = torchvision.models.resnet50(pretrained=True)
elif model_arch == "vgg":
    model = torchvision.models.vgg16(pretrained=True)
elif model_arch == "alexnet":
    model = torchvision.models.alexnet(pretrained=True)  # 227 x 227 input
elif model_arch == "densenet":
    model = torchvision.models.densenet121(pretrained=True)
    

if model_arch == "resnet":
    model.fc = checkpoint["hidden_layers"]
else:
    model.classifier = checkpoint["hidden_layers"]
    

model.load_state_dict(checkpoint["state_dict"])
model.eval()
model.to(device)

# TRANSFORMING OUR IMAGE TO USE IT TO MAKE PREDICTIONS

input_image_size = (227,227) if checkpoint["model_arch"] == "alexnet" else (224,224)
transform_test = transforms.Compose([transforms.Resize(size=input_image_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
image = Image.open(args.image_path)
normalised_tensor_image = transform_test(image)
normalised_tensor_image = normalised_tensor_image.to(device)

# MAKING PREDICTIONS
# Initialsing softmax to make probability distribution
softmax = torch.nn.Softmax(dim=1)
pred = softmax(model.forward(normalised_tensor_image.view(1,3,224,224)))
prob, classes = pred.topk(args.top_k)
# Getting Flower names from the prediction we just produced

# LOADING THE CLASSES_TO_NAMES DICTIONARY
with open(args.category_names, 'r') as f:
    category_to_name = json.load(f)
idx_to_classes = {str(k):str(i) for i,k in checkpoint["classes_to_idx"].items()}

prediction_flower_name = [category_to_name[idx_to_classes[str(int(i))]] for i in classes[0]]

print("======== PREDICTION OUTPUT =========\n")
print("Predicted Flower Names: ")
for name,prob in zip(prediction_flower_name,prob[0]):
    print(name + " ( "+str(float(prob*100))[:4]+"% confidence )")
print(category_to_name["11"])

