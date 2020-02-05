import os
import random
import sys
import time

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageChops
from torchvision.utils import make_grid

from models import Generator
from models_files_settings import ModelsFilesSettings


class ImageGeneration:
    def __init__(self):
        self.count_generated_images = 0
        self.last_time_random_mode_changed_style = 0
        self.seconds_delay_for_random_mode_to_wait_before_changing_style = 20

        self.is_in_random_mode = False
        self.current_used_style_name = None
        self.current_selected_style_type_or_name = None
        self.has_style_type_just_changed = False

        self.input_and_output_image_size = 256
        self.num_color_dims_input_data = 3
        self.num_color_dims_output_data = 3
        # 3 colors channels (red, green blue)

        self.image_transforms = [
            transforms.Resize(int(self.input_and_output_image_size * 1.12), Image.BICUBIC),
            transforms.RandomCrop(self.input_and_output_image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"{'GPU' if torch.cuda.is_available() else 'CPU'} is being used")
        self.map_location = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.currently_used_netG_A2B = None
        self.netG_A2B_dict_for_all_styles = dict()
        for key_style_model_folderpath, values_style_model_folderpath in ModelsFilesSettings.statedicts_folderpaths_per_style.items():
            if key_style_model_folderpath not in ModelsFilesSettings.iter_number_per_style:
                raise Exception(f"{key_style_model_folderpath} key is missing from the ModelsFilesSettings.iter_number_per_style")

            iter_number = ModelsFilesSettings.iter_number_per_style[key_style_model_folderpath]
            netG_A2B_statedict_filepath = os.path.join(values_style_model_folderpath, f"iter-{iter_number}_netG_A2B.pth")

            if not os.path.isfile(netG_A2B_statedict_filepath):
                raise Exception(f"No file was found at {netG_A2B_statedict_filepath}")

            self.netG_A2B_dict_for_all_styles[key_style_model_folderpath] = Generator(self.num_color_dims_input_data, self.num_color_dims_output_data).to(self.device)
            self.netG_A2B_dict_for_all_styles[key_style_model_folderpath].load_state_dict(torch.load(netG_A2B_statedict_filepath, map_location=self.map_location))
            self.netG_A2B_dict_for_all_styles[key_style_model_folderpath].eval()

    def process_input_image(self, image: Image.Image) -> torch.Tensor:
        # Removing potential black borders from the image object
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            image = image.crop(bbox)

        # Resizing the image object
        ratio_to_adjust_img_dims = min(self.input_and_output_image_size / image.size[0], self.input_and_output_image_size / image.size[1])
        img_width = round(image.size[0] * ratio_to_adjust_img_dims)
        img_height = round(image.size[1] * ratio_to_adjust_img_dims)
        img_transforms = transforms.Compose([transforms.Resize((img_height, img_width)), transforms.ToTensor()])

        # Turning the image object into an image tensor
        image = img_transforms(image).to(self.device)
        image = image.unsqueeze(0)
        return image

    def generate_fake_image(self, image_tensor_input) -> torch.Tensor:
        """ Return a tensor of dim 1 (num of image), 3 (rgb colors), X Size of image, Y Size of image"""
        sys.stdout.write(f"\rGenerating image {self.count_generated_images + 1}")

        generated_image_fake_B = 0.5 * (self.currently_used_netG_A2B(image_tensor_input).data + 1.0)
        # generated_image_fake_B is a tensor of dim 1 (num of image), 3 (rgb colors), X Size of image, Y Size of image
        self.count_generated_images += 1

        if not isinstance(generated_image_fake_B, torch.Tensor):
            raise Exception("The generated image was not in the form of a tensor !")

        # We return the first index of the tensor, because the tensor is a list of images, and since
        # we generated only one image, we want to get only the tensor values for this image
        return generated_image_fake_B[0]

    @staticmethod
    def process_generated_image_tensor(image_tensor: torch.Tensor) -> torch.Tensor:
        image_tensor = image_tensor.reshape(image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[0])
        return image_tensor

    @staticmethod
    def turn_image_tensor_to_image_object(image_tensor: torch.Tensor) -> Image.Image:
        grid = make_grid(image_tensor, nrow=12, padding=0, pad_value=0,
                         normalize=False, range=None, scale_each=False)

        # Add 0.5 after un-normalizing to [0, 255] to round to nearest integer
        numpy_treated_array = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(numpy_treated_array)
        return img

    def process_image_source_to_generated(self, image_source: Image.Image) -> Image.Image:
        img_tensor = self.process_input_image(image=image_source)
        generated_image_tensor = self.generate_fake_image(img_tensor)
        processed_generated_image = self.turn_image_tensor_to_image_object(generated_image_tensor).resize((1980, 1080))
        return processed_generated_image

    def check_random_mode_to_set_style_to_use(self):
        if time.time() > self.last_time_random_mode_changed_style + self.seconds_delay_for_random_mode_to_wait_before_changing_style:
            style_keys_to_chose_from = list(self.netG_A2B_dict_for_all_styles.keys())

            if self.current_used_style_name in style_keys_to_chose_from:
                style_keys_to_chose_from.remove(self.current_used_style_name)
            selected_style_key = random.choice(style_keys_to_chose_from)

            self.currently_used_netG_A2B = self.netG_A2B_dict_for_all_styles[selected_style_key]
            self.current_used_style_name = selected_style_key
            self.has_style_type_just_changed = True
            self.last_time_random_mode_changed_style = time.time()
            print(f"Random mode just changed the current style to {self.current_used_style_name}")
