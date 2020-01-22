import torch
import torchvision
import cv2
from torchvision import transforms

try:
    from .models import Generator
except:
    from models import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
filepath_model_to_convert = "D:/Projets Inoft/Bureau des émotions (génération de portraits)/pycharm_project/emotions_recognition/trained_models/jaune_joie/iter-2501_netG_A2B.pth"

model = Generator(3, 3).to(device)  # 3 is for RGB
model.load_state_dict(torch.load(filepath_model_to_convert, map_location=map_location))
model.eval()

img = cv2.imread("team_robinson_128x128.jpg")
img_transforms = transforms.Compose([transforms.ToTensor()])

# Turning the image object into an image tensor
img = img_transforms(img).to(device)
img = img.unsqueeze(0)

# Run model
generated_image_fake_B = 0.5 * (model(img).data + 1.0)
print(generated_image_fake_B)

