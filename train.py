import argparse
import torch
import torchvision
from torchvision import datasets,transforms
from torch import nn
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# FEW INSTRUCTIONS FOR THE REVIEWER FOR RUNNING THIS CODE
# - Models that are supported are: ["vgg","resnet","densenet","alexnet"]
# - The code was ran and tested on udacity vm and each of the models was able to successfully train without any errors
# ====================================================================================================


def train():
    
    # GETTING THE COMMAND LINE ARGUMENTS INPUT

    parser = argparse.ArgumentParser()
    parser.add_argument("dir",default="flowers",type=str)
    parser.add_argument("--save_dir",default="checkpoints",type=str)
    parser.add_argument("--arch",default="resnet",type=str)
    parser.add_argument("--learning_rate",default=0.001,type=float)
    parser.add_argument("--hidden_units",default=512,type=int)
    parser.add_argument("--epochs",default=5,type=int)
    parser.add_argument("--gpu",action='store_true')

    args = parser.parse_args()
    print(args.dir)
    prefered_device = "cuda" if args.gpu else "cpu"
    
    # IF CUDA IS AVAILABLE AND USER WANTS TO USE GPU THEN WE GOING TO SET DEVICE TO CUDA
    if torch.cuda.is_available() and prefered_device=="cuda":
        device = "cuda"
    else:
        device = "cpu"
        
    # Specifying the data directories
    data_dir = args.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Since different models have different input sizes so making it flexible according to model arch
    if args.arch == "alexnet":
        input_image_size = (227,227)
    else:
        input_image_size = (224,224)

    # Transforms for the training, validation, and testing sets
    transform_train = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.Resize(size=input_image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ])
    transform_test = transforms.Compose([transforms.Resize(size=input_image_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ])

    # Loading the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=transform_test)

    # DEFINING THE DATALOADERS FROM THE DATASETS

    trainloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=64,shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,batch_size=64,shuffle=True)

    # LOADING OUR PRETRAINED MODEL

    model_arch = args.arch
    if model_arch == "resnet":
        model = torchvision.models.resnet50(pretrained=True)
    elif model_arch == "vgg":
        model = torchvision.models.vgg16(pretrained=True)
    elif model_arch == "alexnet":
        model = torchvision.models.alexnet(pretrained=True)  # 227 x 227 input
    elif model_arch == "densenet":
        model = torchvision.models.densenet121(pretrained=True)

    # FREEZING THE WEIGHTS


    for layers in model.parameters():
        layers.requires_grad = False

    last_layer_input_nodes = {
        "vgg":25088,
        "alexnet":9216,
        "resnet":2048,
    }

    # MODIFYING THE LAST LAYER TO CUSTOM OUTPUT CLASSES
    if model_arch == "densenet":
        model.classifier = nn.Sequential(
                                  nn.Linear(1024,args.hidden_units),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(args.hidden_units,len(train_dataset.class_to_idx)))

    elif model_arch == "alexnet" or model_arch == "vgg":
        model.classifier = nn.Sequential(
                                  nn.Dropout(p=0.5),
                                  nn.Linear(last_layer_input_nodes[model_arch],4096),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.5),
                                  nn.Linear(4096,4096),
                                  nn.ReLU(),
                                  nn.Linear(4096,args.hidden_units),
                                  nn.ReLU(),
                                  nn.Linear(args.hidden_units,len(train_dataset.class_to_idx)))

    elif model_arch == "resnet":
        model.fc = nn.Sequential(
                              nn.Linear(last_layer_input_nodes[model_arch],args.hidden_units),
                              nn.ReLU(),
                              nn.Dropout(p=0.2),
                              nn.Linear(args.hidden_units,len(train_dataset.class_to_idx)))

    # SPECIFYING THE LOSS AND OPTIMIZERS FOR THE MODEL

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.fc.parameters() if model_arch == "resnet" else                               model.classifier.parameters(),lr=args.learning_rate)

    # TRAINING OUR MODEL !!!!

    softmax = nn.Softmax(dim=1)
    epochs = args.epochs
    train_loss = []
    validation_loss = []
    model.to(device)

    for e in range(epochs):

        total_loss = 0
        total_validation_loss = 0
        total_correct_classifications = 0

        # For Printing Ouput 
        j = 0
        print("Epoch "+str(e+1)+" : ")
        print(str(j)+"% ["+"|"*j+" "*(100-j)+"]"+" train loss: 0.00",end="\r")
        # ===================

        # OUR TRAINING STEP

        for step,(images,labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            pred = model.forward(images)
            loss = criterion(pred,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss

            # FOR PRINTING PROGRESS BAR
            j = int((step / len(trainloader))*100)+1
            print(str(j)+"% ["+"|"*j+" "*(100-j)+"]"+" train loss: "+str(float(loss))[:4],end="\r")
            # ==========================

        print("\n")
        print("Evaluating...")

        # EVALUATING OUR MODEL ON VALIDATION IMAGES SET

        model.eval()
        for images,labels in validation_loader:

            # Getting the validation loss on validation dataset
            images = images.to(device)
            labels = labels.to(device)
            pred = model.forward(images)
            loss = criterion(pred,labels)
            total_validation_loss += loss

            # Calculating Accuracy
            prob, classes = pred.topk(1)
            got_correct = labels == classes.view(*labels.shape)
            no_of_correct_classifications = torch.sum(got_correct.type(torch.FloatTensor))
            total_correct_classifications += no_of_correct_classifications

        train_loss.append(total_loss/len(trainloader.dataset))
        validation_loss.append(total_validation_loss/len(validation_loader.dataset))

        print("Training Loss: "+str(train_loss[-1]))
        print("Validation Loss: "+str(validation_loss[-1]))
        print("Accuracy: "+str(int( (total_correct_classifications/len(validation_loader.dataset))*100 ))+"%")
        model.train()


    # AT LAST! WE ARE TESTING ACCURACY OF OUR MODEL ON UNSEEN TEST IMAGES SET AND PRINTING IT!!

    print("\n")
    print("CALCULATING MODEL ACCURACY ON TEST IMAGES SET...")

    total_correct_classifications = 0

    for test_images, test_labels in test_loader:
        test_images.to(device)
        test_labels.to(device)
        pred = softmax(model.forward(images))
        prob, classes = pred.topk(1)
        got_correct = labels == classes.view(*labels.shape)
        no_of_correct_classifications = torch.sum(got_correct.type(torch.FloatTensor))
        print(no_of_correct_classifications)
        total_correct_classifications += no_of_correct_classifications

    accuracy = total_correct_classifications / len(test_loader.dataset)
    print("TEST SET ACCURACY RESULTS:")
    print("The model classified "+str(total_correct_classifications)+" out of "+str(len(test_loader.dataset))+" images            correctly!")
    print("Accuracy: "+str(int(accuracy*100))+"%")

    # NOW WE ARE GOING TO SAVE WHAT WE JUST TRAINED (I MEAN THE MODEL WEIGHTS) !!! 

    checkpoint_data = {
        "input_shape":input_image_size,
        "output_shape":102,
        "hidden_layers":model.fc if model_arch=="resnet" else model.classifier,
        "state_dict":model.state_dict(),
        "classes_to_idx":train_dataset.class_to_idx,
        "optimizer_state_dict":optimizer.state_dict(),
        "model_arch":model_arch,

    }

    torch.save(checkpoint_data, args.save_dir+"/model.pth")

    print("Model got saved successfully at "+args.save_dir+"/model.pth")


if __name__ == "__main__":
    train()
# -----------------------------------------------------------------------------------------------

#          END OF OUR TRAINING CODE 
#           THANK YOU
