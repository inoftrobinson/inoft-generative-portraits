import os
import sys
import time

from PIL import Image
from matplotlib import use as set_plt_backend
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt


class DisplayWindowHandlers:
    def __init__(self, set_window_fullscreen: bool):
        self.current_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.emotion_thumbnail_image = None

        set_plt_backend("tkagg")

        # Removing the toolbar (need to be done before creating the fig)
        plt.rcParams["toolbar"] = "None"

        self.fig, self.ax = plt.subplots(1, figsize=(16, 9), constrained_layout=False)

        self.fig.canvas.set_window_title("Inoft Portraits - Bureau des émotions")

        # Remove the y and x axis (need to be done after creating the fig)
        plt.axis("off")

        # Disable any margin by sticking the subplots in the corners of the window (need to be done after creating the fig)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        if set_window_fullscreen:
            # Display the window in full screen
            mng = plt.get_current_fig_manager()
            mng.full_screen_toggle()

        # Make the fig take the all window (because it might not take the entire resolution of the screen, needs to be the last operation on the fig to avoid resizing issues)
        self.fig.set_size_inches(19.20, 10.80 + 0.02)  # 1 inch is 100 pixels
        # For some reasons i'm losing 0.02 inches (2 pixels) when i'm setting the y size, so i'm adding those 2 pixels

        # self.fig.canvas.mpl_connect("close_event", self.close_program())

    def close_program(event):
        print("Quiting program")
        # plt.close()
        sys.exit()

    def set_emotion_thumbnail_image(self, current_used_style_name: str) -> None:
        image_path = os.path.join(self.current_dir_path, "icons_images", f"{current_used_style_name}.png")
        if os.path.isfile(image_path):
            try:
                image = Image.open(image_path)
                image.thumbnail((128, 128), Image.ANTIALIAS)  # resize image in-place
                self.emotion_thumbnail_image = image
            except Exception as error:
                # Sometimes, when trying to open the image file, there might be a crash, if that's the case, we just skip
                # the setting of the thumbnail for the loop and let it set itself to None, and do not make crash the program.
                print(f"Non-crashing error while trying to open the processed generated image received from the ftp at {time.time()}s")
        else:
            self.emotion_thumbnail_image = None

    def put_emotion_thumbnail_on_figure(self, image_plt_object: AxesImage) -> Figure:
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

            plt_thumbnail_image_object = self.fig.figimage(self.emotion_thumbnail_image, emotion_thumbnail_width_pixel_position, emotion_thumbnail_height_pixel_position)
            return plt_thumbnail_image_object

        return None

    def show_image(self, image_object_to_show):
        plt_image_object = self.ax.imshow(image_object_to_show, interpolation="nearest")
        # self.ax.set_aspect("auto")
        # We need to set the aspect to auto on every frame,
        # otherwise the image might not fill up to entire screen.
        return plt_image_object

    def save_fig(self, filepath: str) -> None:
        plt.savefig(filepath)

    def loop_paue(self) -> None:
        plt.pause(0.01)

