import torch
from torchvision import transforms
from models import Generator
"""
models_files = glob.glob(values_style_model_folderpath + "/*.pth")
                    count_model_files = 0
                    for model_file in models_files:
                        if "netG_A2B.pth" in model_file:
                            count_model_files += 1
                            self.netG_A2B_dict_for_all_styles.append({"model": Generator(opt.input_nc, opt.output_nc).to(self.device), "filename": model_file})
                            self.netG_A2B_dict_for_all_styles[count_model_files - 1]["model"].load_state_dict(torch.load(model_file, map_location=self.map_location))
                            self.netG_A2B_dict_for_all_styles[count_model_files - 1]["model"].eval()
"""
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
folderpath_of_models_to_convert = "F:/Bureau des émotions/trained_models/bleu-clair-surprise" #/iter-2501_netG_A2B.pth"

model = Generator(3, 3).to(device)  # 3 is for RGB
model.load_state_dict(torch.load(folderpath_of_models_to_convert, map_location=map_location))
model.eval()

img = cv2.imread("team_robinson_128x128.jpg")
img_transforms = transforms.Compose([transforms.ToTensor()])

# Turning the image object into an image tensor
img = img_transforms(img).to(device)
img = img.unsqueeze(0)

# Run model
generated_image_fake_B = 0.5 * (model(img).data + 1.0)
print(generated_image_fake_B)

