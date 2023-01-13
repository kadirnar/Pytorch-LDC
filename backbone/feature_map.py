import torch
from modelB4 import LDC
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

img=cv2.imread("8068.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img=np.array(img)
img=transform(img)
img=img.unsqueeze(0)

checkpoint_path = 'checkpoints/BRIND/16/16_model.pth'


def load_model(checkpoint_path):
    model = LDC()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model(checkpoint_path)

def get_conv_layers(model):
    conv_layers = []
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            conv_layers.append(layer)
    return conv_layers

conv_layers = get_conv_layers(model)
results = [conv_layers[0](img)]
for i in range(1, 3):
    results.append(conv_layers[i](results[-1]))
outputs = results


for num_layer in range(len(outputs)):
    plt.figure(figsize=(50, 10))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print("Layer ",num_layer+1)
    for i, filter in enumerate(layer_viz):
        if i == 16: 
            break
        plt.subplot(2, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    plt.show()
    plt.close()
    
   
