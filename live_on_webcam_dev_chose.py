import glob
import random
import os
import sys

import cv2
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

activate_source_sender_result_receiver = False
activate_source_receiver_result_sender = False
if activate_source_sender_result_receiver is True and activate_source_receiver_result_sender is True:
    raise Exception("The 2 modes cannot be active at the same time.")

generated_image_resizing_height_dim = 720
generated_image_resizing_width_dim = 480

if activate_source_receiver_result_sender is not True:
    # We need the video_stream, unless we receive the source. So if we have no communication mode activated,
    # or that we are using the source sender and not the source receiver, we need to the video stream.
    video_stream = cv2.VideoCapture(0)

if activate_source_sender_result_receiver is True:
    import ftp_communication_source_sender_result_receiver
    thread_class_upload_image_source_from_webcam = ftp_communication_source_sender_result_receiver.ThreadUploadImageSourceFromWebcam(video_stream=video_stream)
    thread_class_save_generated_image_from_ftp = ftp_communication_source_sender_result_receiver.ThreadSaveGeneratedImageFromFtp()
elif activate_source_receiver_result_sender is True:
    import ftp_communication_source_receiver_result_sender
    thread_class_save_image_source_from_ftp = ftp_communication_source_receiver_result_sender.ThreadSaveImageSourceFromFtp()
    thread_class_upload_generated_image = ftp_communication_source_receiver_result_sender.ThreadUploadGeneratedImage()

def close_program(event):
    print("Quiting program")
    plt.close()
    sys.exit()

fig, ax = None, None
control_fig, control_ax = None, None
import matplotlib.pyplot as plt
use_control_window = False

# We create the window here, because after all of our imports, we might have issues create the windows and plots
def create_matplotlib_window():
    # Make the fig accessible outside this function with global variables
    global fig, ax

    # Removing the toolbar (need to be done before creating the fig)
    plt.rcParams["toolbar"] = "None"

    # Creating the figure
    fig, ax = plt.subplots(1, figsize=(16, 9), constrained_layout=False)

    # Set window title and icon on the matplotlib window
    fig.canvas.set_window_title("Inoft Portraits - Bureau des émotions")

    # When the window is closed, stop the ai, otherwise the ai would keep running even if the window is closed
    fig.canvas.mpl_connect("close_event", close_program)

    # Remove the y and x axis (need to be done after creating the fig)
    plt.axis("off")

    # Disable any margin by sticking the subplots in the corners of the window (need to be done after creating the fig)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Display the window in full screen
    mng = plt.get_current_fig_manager()
    # mng.full_screen_toggle()

    # Make the fig take the all window (because it might not take the entire resolution of the screen, needs to be the last operation on the fig to avoid resizing issues)
    fig.set_size_inches(1920 / 100, (1080 / 100) + 0.02)  # 1 inch is 100 pixels
    # For some reasons i'm losing 0.02 inches (2 pixels) when i'm setting the y size, so i'm adding those 2 pixels

    if use_control_window:
        global control_fig, control_ax
        plt.rcParams["toolbar"] = "None"
        control_fig, control_ax = plt.subplots(1, figsize=(16, 9), constrained_layout=False)
        control_fig.canvas.set_window_title("Inoft Portraits - Bureau des émotions - Contrôle")
        control_fig.canvas.mpl_connect("close_event", close_program)
        control_fig.set_size_inches(19.20, 10.80 + 0.02)  # 1 inch is 100 pixels
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    
if __name__ == "__main__":
    # We also create the window only if we are the main scripts, since secondary scripts
    # can import this script yet we do not want to create the window multiple times
    create_matplotlib_window()
    print(plt.get_backend())

import time
from datetime import datetime

if activate_source_sender_result_receiver is False:
    import torchvision.transforms as transforms
    from torchvision.utils import save_image, make_grid
    from torch.utils.data import DataLoader
    from torch.autograd import Variable
    import torch
from PIL import Image, ImageGrab, ImageChops, ImageFile

from models import Generator
from datasets import ImageDataset


# region Args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batchSize", type=int, default=1, help="size of the batches")
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--size', type=int, default=800, help='size of the data (squared assumed)') # Default = 256
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)
# endregion

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = "C:/Users/CreativeAI/Documents/Bureau des émotions/outputs/output"

style_folder_name = "clone"
models_statedict_folderpaths_per_style = {
    "surprise": os.path.join(base_dir, "bleu-clair-surprise"),
    "tristesse": os.path.join(base_dir, "bleu-fonce-tristesse"),
    "joie": os.path.join(base_dir, "jaune-joie"),
    "excitation": os.path.join(base_dir, "orange-excitation"),
    "colere": os.path.join(base_dir, "rouge-colere_250-faces"),
    "peur": os.path.join(base_dir, "vert-bleu-peur"),
    "attirance": os.path.join(base_dir, "vert-clair-amour"),
    "degout": os.path.join(base_dir, "violet-degout"),
    "star-wars": "C:/Users/CreativeAI/Documents/Bureau des émotions/outputs/output_star-wars/general"
}
model_multiplicator = 3
general_index = model_multiplicator * 25
excitation_index = model_multiplicator * 26
models_iter_number_per_style = {
    "surprise": general_index,
    "tristesse": general_index,
    "joie": general_index,
    "excitation": excitation_index,
    "colere": general_index,
    "peur": general_index,
    "attirance": general_index,
    "degout": general_index,
    "star-wars": general_index,
}

overriden_style_name = "star-wars"

KEY_UNPROCESSED = "unprocessed"
KEY_GENERATED = "generated"

class NetworkSystem:
    if activate_source_sender_result_receiver is False:
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    else:
        Tensor = None
        # We need to create a Tensor variable, since some of the functions return a Tensor
        # output, if the type of their outputs is not declared, it will cause an exception.

    def __init__(self):
        if activate_source_sender_result_receiver is False:
            # Device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"{'GPU' if torch.cuda.is_available() else 'CPU'} is being used")
            self.map_location = "cuda:0" if torch.cuda.is_available() else "cpu"

            # region Networks
            self.currently_used_netG_A2B = None
            self.netG_A2B_dict_for_all_styles = dict()
            # self.netG_B2A_dict_for_all_styles = dict()
            for key_style_model_folderpath, values_style_model_folderpath in models_statedict_folderpaths_per_style.items():
                if key_style_model_folderpath == overriden_style_name:
                    if key_style_model_folderpath not in models_iter_number_per_style:
                        raise Exception(f"{key_style_model_folderpath} key is missing from the models_iter_number_per_style")

                    iter_number = models_iter_number_per_style[key_style_model_folderpath]
                    netG_A2B_statedict_filepath = os.path.join(values_style_model_folderpath, f"iter-{iter_number}_netG_A2B.pth")
                    # netG_B2A_statedict_filepath = os.path.join(values_style_model_folderpath, f"iter-{iter_number}_netG_B2A.pth")

                    if not os.path.isfile(netG_A2B_statedict_filepath):
                        raise Exception(f"No file was found at {netG_A2B_statedict_filepath}")
                        raise Exception(f"No file was found at {netG_A2B_statedict_filepath}")
                    # if not os.path.isfile(netG_B2A_statedict_filepath):
                    #     raise Exception(f"No file was found at {netG_A2B_statedict_filepath}")

                    self.netG_A2B_dict_for_all_styles = list()

                    models_files = glob.glob(values_style_model_folderpath + "/*.pth")
                    count_model_files = 0
                    for model_file in models_files:
                        if "netG_A2B.pth" in model_file:
                            count_model_files += 1
                            self.netG_A2B_dict_for_all_styles.append({"model": Generator(opt.input_nc, opt.output_nc).to(self.device), "filename": model_file})
                            self.netG_A2B_dict_for_all_styles[count_model_files - 1]["model"].load_state_dict(torch.load(model_file, map_location=self.map_location))
                            self.netG_A2B_dict_for_all_styles[count_model_files - 1]["model"].eval()

                    # self.netG_B2A_dict_for_all_styles[key_style_model_folderpath] = Generator(opt.input_nc, opt.output_nc).to(self.device)
                    # self.netG_B2A_dict_for_all_styles[key_style_model_folderpath].load_state_dict(torch.load(netG_B2A_statedict_filepath, map_location=self.map_location))
                    # self.netG_B2A_dict_for_all_styles[key_style_model_folderpath].eval()
            # endregion

            # region Inputs
            self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
            self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
            # endregion

        # region Resolutions, counts and holders
        self.image_size = 800

        self.last_time_random_mode_changed_style = 0
        self.seconds_delay_for_random_mode_to_wait_before_changing_style = 3

        self.index_last_selected_style = None
        self.count_generated_images = 0
        self.is_in_random_mode = False
        self.current_used_style_name = None
        self.current_selected_style_type_or_name = None
        self.has_style_type_just_changed = False
        self.need_to_save_pictures = False
        self.additional_text_to_use_in_filenames = None

        self.num_seconds_before_removing_saved_unprocessed_image = 10
        self.last_images = dict()
        self.emotion_thumbnail_image = None

        self.current_dir_path = self.get_current_dir()
        self.filepath_temp_image_generated_to_send = os.path.join(self.current_dir_path, "temp", "image_generated_to_send.jpg")
        # endregion

        # Dataset loader
        if activate_source_sender_result_receiver is False:
            transforms_ = [
                transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                transforms.RandomCrop(opt.size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

        #region Creating output dirs
        if not os.path.exists('output/A'):
            os.makedirs('output/A')
        if not os.path.exists('output/B'):
            os.makedirs('output/B')

        self.FOLDER_NAME_SAVED_UNPROCESSED_IMAGES = "saved_images"
        #endregion

    def remove_saved_unprocessed_image_older_than(self, older_than_num_seconds: int):
        keys_images_to_remove = list()

        for time_saved_image_key in self.last_images.keys():
            float_current_time = float(time.time()) # Just to make sure the time is a float if the module came to be changed
            float_time_saved_image = float(time_saved_image_key)
            if float_current_time > (float_time_saved_image + older_than_num_seconds):
                keys_images_to_remove.append(time_saved_image_key)

        for key_image_to_remove in keys_images_to_remove:
            self.last_images.pop(key_image_to_remove)

    def week_number_of_month(self, date_value):
        # Source : https://www.mytecbits.com/internet/python/week-number-of-month
        return date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1

    def get_and_make_dir_to_save_images_for_current_date(self):
        """ Create one directory per month and create 4 directory for every week """
        base_folderpath = os.path.join(self.current_dir_path, self.FOLDER_NAME_SAVED_UNPROCESSED_IMAGES)
        if not os.path.exists(base_folderpath):
            os.makedirs(base_folderpath)

        now_datetime = datetime.now()
        current_year_folderpath = os.path.join(base_folderpath, str(now_datetime.year))
        if not os.path.exists(current_year_folderpath):
            os.makedirs(current_year_folderpath)

        date_given = datetime.today().date()
        week_number_of_month = self.week_number_of_month(date_given)
        current_week_number_folderpath = os.path.join(current_year_folderpath, f"Semaine-{week_number_of_month}")
        if not os.path.exists(current_week_number_folderpath):
            os.makedirs(current_week_number_folderpath)

        return current_week_number_folderpath

    def save_recents_images(self):
        print(f"Saving {len(self.last_images) * 2} images")
        # We multiply the length by 2 since we save the unprocessed and the generated image

        for time_saved_image_key, saved_images_dict in self.last_images.items():
            timestamp_without_milliseconds, timestamp_milliseconds_only = str(time_saved_image_key).split(".", maxsplit=1)
            formatted_date_string = datetime.utcfromtimestamp(int(timestamp_without_milliseconds)).strftime("%Y-%m-%d_%Hh%Mm%Ss") + f"-{timestamp_milliseconds_only}millios"

            if isinstance(self.additional_text_to_use_in_filenames, str) and self.additional_text_to_use_in_filenames.replace(" ", "") != "":
                # If we have some additional text to use in the filenames, we remove the potential special chars from the text to make sure windows do not explode
                formatted_date_string = f"{''.join(char for char in self.additional_text_to_use_in_filenames if char.isalnum())}_{formatted_date_string}"

            if isinstance(saved_images_dict, dict):
                if KEY_UNPROCESSED in saved_images_dict.keys():
                    unprocessed_image_object = saved_images_dict[KEY_UNPROCESSED]
                    if isinstance(unprocessed_image_object, Image.Image):
                        unprocessed_image_filename = f"{formatted_date_string}_unprocessed.jpg"
                        unprocessed_image_object.save(os.path.join(self.get_and_make_dir_to_save_images_for_current_date(), unprocessed_image_filename))

                if KEY_GENERATED in saved_images_dict.keys():
                    generated_image_object = saved_images_dict[KEY_GENERATED]
                    if isinstance(generated_image_object, Image.Image):
                        generated_image_filename = f"{formatted_date_string}_generated.jpg"
                        generated_image_object.save(os.path.join(self.get_and_make_dir_to_save_images_for_current_date(), generated_image_filename))

    def check_random_mode_to_set_style_to_use(self):
        if time.time() > self.last_time_random_mode_changed_style + self.seconds_delay_for_random_mode_to_wait_before_changing_style:
            self.index_last_selected_style = 0 if self.index_last_selected_style is None else self.index_last_selected_style + 1
            if self.index_last_selected_style + 1 > len(self.netG_A2B_dict_for_all_styles):
                self.index_last_selected_style = 0

            self.currently_used_netG_A2B = self.netG_A2B_dict_for_all_styles[self.index_last_selected_style]["model"]
            self.current_used_style_name = overriden_style_name
            self.has_style_type_just_changed = True
            self.last_time_random_mode_changed_style = time.time()
            print(f"Random mode just changed the current style to filename : {self.netG_A2B_dict_for_all_styles[self.index_last_selected_style]['filename']}")

    def process_input_image(self, image: Image.Image) -> Tensor:
        # Removing potential black borders from the image object
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            image = image.crop(bbox)

        # Resizing the image object
        ratio_to_adjust_img_dims = min(self.image_size / image.size[0], self.image_size / image.size[1])
        img_width = round(image.size[0] * ratio_to_adjust_img_dims)
        img_height = round(image.size[1] * ratio_to_adjust_img_dims)
        img_transforms = transforms.Compose([transforms.Resize((img_height, img_width)), transforms.ToTensor()])

        # Turning the image object into an image tensor
        image = img_transforms(image).to(self.device)
        image = image.unsqueeze(0)
        return image

    def generate_fake_image(self, image_tensor_input) -> Tensor:
        """ Return a tensor of dim 1 (num of image), 3 (rgb colors), X Size of image, Y Size of image"""
        # sys.stdout.write(f"/rGenerating image {self.count_generated_images + 1}")

        generated_image_fake_B = 0.5 * (self.currently_used_netG_A2B(image_tensor_input).data + 1.0)
        # generated_image_fake_B is a tensor of dim 1 (num of image), 3 (rgb colors), X Size of image, Y Size of image
        self.count_generated_images += 1
    
        if not isinstance(generated_image_fake_B, self.Tensor):
            raise Exception("The generated image was not in the form of a tensor !")
    
        # We return the first index of the tensor, because the tensor is a list of images, and since
        # we generated only one image, we want to get only the tensor values for this image
        return generated_image_fake_B[0]
    
    def process_generated_image_tensor(self, image_tensor: Tensor) -> Tensor:
        image_tensor = image_tensor.reshape(image_tensor.shape[1],
                                            image_tensor.shape[2],
                                            image_tensor.shape[0])
        return image_tensor
    
    def turn_image_tensor_to_image_object(self, image_tensor: Tensor) -> Image.Image:
        grid = make_grid(image_tensor, nrow=12, padding=0, pad_value=0,
                         normalize=False, range=None, scale_each=False)
    
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        # img_numpy_array = image_tensor.to("cpu", torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        return img
    
    def save_image_tensor_to_image_file(self, image_tensor: Tensor):
        # This function is not called save_image, since save_image is an util function of torchvision
        save_image(image_tensor, f"output/B/{self.count_generated_images}.png")
    
    def get_current_dir(self) -> str:
        return os.path.dirname(os.path.abspath(__file__))

    def set_emotion_thumbnail_image(self, current_used_style_name: str):
        image_path = os.path.join(self.current_dir_path, "icons_images", f"{current_used_style_name}.png")
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            image.thumbnail((128, 128), Image.ANTIALIAS)  # resizes image in-place
            return image
        else:
            return None

    def put_emotion_thumbnail_on_figure(self, image_plt_object: AxesImage, plt_figure: Figure):
        generated_image_left_start_in_fig_width = image_plt_object.clipbox.intervalx.min()
        generated_image_right_end_in_fig_width = image_plt_object.clipbox.intervalx.max()
        generated_image_width_pixel_size = generated_image_right_end_in_fig_width - generated_image_left_start_in_fig_width

        generated_image_top_in_fig_height = image_plt_object.clipbox.intervaly.max()
        generated_image_bottom_in_fig_height = image_plt_object.clipbox.intervaly.min()
        generated_image_height_pixel_size = generated_image_top_in_fig_height - generated_image_bottom_in_fig_height

        if self.emotion_thumbnail_image is not None:
            emotion_thumbnail_width = self.emotion_thumbnail_image.size[0]
            emotion_thumbnail_height = self.emotion_thumbnail_image.size[1]
            emotion_thumbnail_percentage_value_height_margin = 4
            emotion_thumbnail_percentage_value_width_margin = 4

            emotion_thumbnail_height_pixel_margin = generated_image_height_pixel_size * (emotion_thumbnail_percentage_value_height_margin / 100)
            emotion_thumbnail_height_pixel_position = (generated_image_top_in_fig_height - emotion_thumbnail_height) - emotion_thumbnail_height_pixel_margin
            # We remove the thumbnail height and the margin from the fig height because the bottom position is 0 of height and we want the thumbnail to go down from the top

            emotion_thumbnail_width_pixel_margin = generated_image_width_pixel_size * (emotion_thumbnail_percentage_value_width_margin / 100)
            emotion_thumbnail_width_pixel_position = generated_image_left_start_in_fig_width + emotion_thumbnail_width_pixel_margin
            # We add the margin from the fig width because the left start position is 0 of width and we want to move the thumbnail to the right and we do not need to also add the thumbnail width

            plt_thumbnail_image_object = plt_figure.figimage(self.emotion_thumbnail_image, emotion_thumbnail_width_pixel_position, emotion_thumbnail_height_pixel_position)
            return plt_thumbnail_image_object
        return None

    def process_image_source_to_generated(self, image_source: Image.Image) -> Image.Image:
        img_tensor = self.process_input_image(image=image_source)
        generated_image_tensor = self.generate_fake_image(img_tensor)
        processed_generated_image = (self.turn_image_tensor_to_image_object(generated_image_tensor)
                                     .resize((generated_image_resizing_height_dim, generated_image_resizing_width_dim)))
        return processed_generated_image

    def start_network_loop(self):
        if activate_source_sender_result_receiver is False:
            import api_communication
            api_communication.trigger_async_api_listener_loop(parent_networkSystem=self, channel_id=api_communication.CHANNEL_ID)

        if activate_source_receiver_result_sender is True:
            # Do not use an elif here, because usually, if the source receiver result sender is activate, the script
            # need to get infos from the API. Its only if he is the source sender result receiver that he do not need to.
            thread_class_save_image_source_from_ftp.start()
            thread_class_upload_generated_image.start()
        elif activate_source_sender_result_receiver is True:
            thread_class_upload_image_source_from_webcam.start()
            thread_class_save_generated_image_from_ftp.start()

        index_image = 0
        plt_thumbnail_image_object = None
        if use_control_window:
            plt_control_thumbnail_image_object = None

        unprocessed_image_object = None
        while True:
            if activate_source_sender_result_receiver is not True:
                # We want to save the result only on the result sender and not on the result receiver (since the received results might be compressed)
                if (True or self.current_selected_style_type_or_name == "random" or self.current_selected_style_type_or_name is None
                or self.current_selected_style_type_or_name not in self.netG_A2B_dict_for_all_styles.keys()):
                    self.check_random_mode_to_set_style_to_use()
                    # The random function will set the current style name to use for the models
                else:
                    self.current_used_style_name = self.current_selected_style_type_or_name
                    if self.has_style_type_just_changed:
                        self.currently_used_netG_A2B = self.netG_A2B_dict_for_all_styles[self.current_used_style_name]
                if self.has_style_type_just_changed:
                    self.emotion_thumbnail_image = self.set_emotion_thumbnail_image(current_used_style_name=self.current_used_style_name)
                    self.has_style_type_just_changed = False

                # We do not set has_style_type_just_changed to False here. We will set it to False only if we to update the generated image.
                # Like so we make sure we generated an image with the latest style type, even if the image source did not changed.

                if self.need_to_save_pictures is True:
                    self.save_recents_images()
                    self.need_to_save_pictures = False

            need_to_update_generated_image = False
            if activate_source_receiver_result_sender is not True and activate_source_sender_result_receiver is not True:
                # If we are using the source sender result receiver, the handling and sending of the video stream is done in its own thread, not here.
                return_code, crude_frame = video_stream.read()
                if crude_frame is not None:
                    frame = cv2.cvtColor(crude_frame, cv2.COLOR_BGR2RGB)
                    unprocessed_image_object = Image.fromarray(frame)
                    processed_generated_image = self.process_image_source_to_generated(image_source=unprocessed_image_object)
                    need_to_update_generated_image = True
                else:
                    raise Exception("No VideoStream was received from the webcam")

            elif activate_source_receiver_result_sender is True:
                if thread_class_save_image_source_from_ftp.image_source_been_modified_and_not_yet_used is True or self.has_style_type_just_changed is True:
                    # No matter the situation, if the style type has changed, even if the source image has not changed, we need to update the image.
                    if thread_class_upload_generated_image.last_generated_image_has_completed_its_upload is True:
                        # Yet, we need for the last generated image to have completed its upload, before generating a new one.
                        unprocessed_image_object = thread_class_save_image_source_from_ftp.get_image_source()
                        processed_generated_image = self.process_image_source_to_generated(image_source=unprocessed_image_object)
                        thread_class_save_image_source_from_ftp.image_source_been_modified_and_not_yet_used = False
                        need_to_update_generated_image = True

            elif activate_source_sender_result_receiver is True:
                # If we are using the source sender result receiver, the handling and sending of the video stream is done in its own thread, not here.
                if thread_class_save_generated_image_from_ftp.received_new_received_generated_image_not_yet_displayed is True or self.has_style_type_just_changed is True:
                    # No matter the situation, if the style type has changed, even if  the source image has not changed, we need to update the image.
                    ImageFile.LOAD_TRUNCATED_IMAGES = True
                    # Truncated images to True in order to allow partial images to still be loaded
                    processed_generated_image = Image.open(thread_class_save_generated_image_from_ftp.temp_image_processed_complete_filepath)
                    thread_class_save_generated_image_from_ftp.received_new_received_generated_image_not_yet_displayed = False
                    need_to_update_generated_image = True

            if need_to_update_generated_image is True:
                self.remove_saved_unprocessed_image_older_than(older_than_num_seconds=self.num_seconds_before_removing_saved_unprocessed_image)
                self.last_images[str(time.time())] = {KEY_UNPROCESSED: unprocessed_image_object, KEY_GENERATED: processed_generated_image}

                # region Display the results on the matplotlib plot
                # We remove the potential previous thumbnail image from the screen
                if plt_thumbnail_image_object is not None:
                    try:
                        plt_thumbnail_image_object.remove()
                    except Exception:
                        pass

                generated_image_plt_object = ax.imshow(processed_generated_image, interpolation="nearest")
                ax.set_aspect("auto")
                # We need to set the aspect to auto on every frame, otherwise the image might not fill up to entire screen
                plt_thumbnail_image_object = self.put_emotion_thumbnail_on_figure(image_plt_object=generated_image_plt_object, plt_figure=fig)

                if use_control_window:
                    if plt_control_thumbnail_image_object is not None:
                        try:
                            plt_control_thumbnail_image_object.remove()
                        except Exception:
                            pass

                    control_generated_image_plt_object = control_ax.imshow(processed_generated_image, interpolation="nearest")
                    control_ax.set_aspect("auto")
                    plt_control_thumbnail_image_object = self.put_emotion_thumbnail_on_figure(image_plt_object=control_generated_image_plt_object, plt_figure=control_fig)
                # endregion

                if activate_source_receiver_result_sender is True:
                    plt.savefig(self.filepath_temp_image_generated_to_send)
                    image = Image.open(self.filepath_temp_image_generated_to_send)
                    image = image.resize((generated_image_resizing_height_dim, generated_image_resizing_width_dim))
                    image.save(self.filepath_temp_image_generated_to_send)
                    thread_class_upload_generated_image.inform_new_image_to_send()

                index_image += 1

            plt.pause(0.01)
            # Even if the displayed image has not been modified, at every step in the loop we need to call plt.pause,
            # otherwise the previously plotted image will vanish from the plot, and the window might crash.


if __name__ == "__main__":
    networkSystem = NetworkSystem()
    networkSystem.start_network_loop()


