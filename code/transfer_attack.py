from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# NOTE: This is a hack to get around "User-agent" limitations when downloading MNIST datasets
#       see, https://github.com/pytorch/vision/issues/3497 for more information
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)


# data loader -----------------------------------------------------------------#
from datasets import MnistDataset
train_dataset = MnistDataset(training=True, transform=None)
test_dataset = MnistDataset(training=False, transform=None)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=120, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

# enable GPU usage ------------------------------------------------------------#
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda == False:
    print("WARNING: CPU will be used for training.")

# model selection -------------------------------------------------------------#
from models.modelM3 import ModelM3
from models.modelM5 import ModelM5
from models.modelM7 import ModelM7
from models.mymodel import MyModel
from models.mymodel2 import MyModel2
from models.mymodel3 import MyModel3
epsilons = [0, .3]

def load_model(pretrained_model = "ModelM5/0.pth", KERNEL_SIZE = 5):
    if(KERNEL_SIZE == 3):
        model = ModelM3().to(device)
    elif(KERNEL_SIZE == 5):
        model = ModelM5().to(device)
    elif(KERNEL_SIZE == 7):
        model = ModelM7().to(device)
    elif(KERNEL_SIZE == 0):
        model = MyModel().to(device)
    elif(KERNEL_SIZE == -1):
        model = MyModel2().to(device)
    elif(KERNEL_SIZE == -2):
        model = MyModel3().to(device)
    # Load the pretrained model
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    # Set the model in evaluation mode. In this case this is for the Dropout layers
    model.eval()
    return model

KERNEL_SIZE = 3
REF_KERNEL_SIZE = 3

ref_model = load_model(f"ModelM{REF_KERNEL_SIZE}/0.pth", REF_KERNEL_SIZE)
model = load_model(f"ModelM{KERNEL_SIZE}/0.pth", KERNEL_SIZE)


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image



def test( ref_model, model, device, test_loader, epsilon ):
    correct = 0
    adv_examples = []
    tot = 0
    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = ref_model(data)

        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        # print(target.)
        loss = F.nll_loss(output, target)
        ref_model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += (torch.flatten(final_pred) == target).sum()
        """
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        """
        # print(i)
    final_acc = correct/float(len(test_loader)*100)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    return final_acc, adv_examples

accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc, ex = test(ref_model, model, device, test_loader, eps)
    accuracies.append(acc)
    examples.append(ex)

"""
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
"""

"""
# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
"""
