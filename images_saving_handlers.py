from datetime import datetime
import os
import time
from PIL import Image
import utils


class ImagesSavingHandlers:
    FOLDER_NAME_SAVED_IMAGES = "saved_images"
    KEY_UNPROCESSED = "unprocessed"
    KEY_GENERATED = "generated"

    def __init__(self):
        self.current_dir_path = os.path.dirname(os.path.abspath(__file__))
        self.last_images = dict()

        self.need_to_save_pictures = False
        self.additional_text_to_use_in_filenames = None
        self.num_seconds_before_removing_saved_unprocessed_image = 10

    def remove_too_old_saved_unprocessed_image(self) -> None:
        keys_images_to_remove = list()

        for time_saved_image_key in self.last_images.keys():
            float_current_time = float(time.time())
            # Just to make sure the time is a float if the module came to be changed

            float_time_saved_image = float(time_saved_image_key)
            if float_current_time > (float_time_saved_image + self.num_seconds_before_removing_saved_unprocessed_image):
                keys_images_to_remove.append(time_saved_image_key)

        for key_image_to_remove in keys_images_to_remove:
            self.last_images.pop(key_image_to_remove)

    def get_and_make_dir_to_save_images_for_current_date(self) -> str:
        """ Create one directory per year per month and create 4 directory as the 4 weeks of a month """

        base_folderpath = os.path.join(self.current_dir_path, self.FOLDER_NAME_SAVED_IMAGES)
        if not os.path.exists(base_folderpath):
            os.makedirs(base_folderpath)

        now_datetime = datetime.now()
        current_year_folderpath = os.path.join(base_folderpath, str(now_datetime.year))
        if not os.path.exists(current_year_folderpath):
            os.makedirs(current_year_folderpath)

        current_month_folderpath = os.path.join(current_year_folderpath, str(now_datetime.month))
        if not os.path.exists(current_month_folderpath):
            os.makedirs(current_month_folderpath)

        date_given = datetime.today().date()
        week_number_of_month = utils.week_number_of_month(date_given)
        current_week_number_folderpath = os.path.join(current_month_folderpath, f"Semaine-{week_number_of_month}")
        if not os.path.exists(current_week_number_folderpath):
            os.makedirs(current_week_number_folderpath)

        return current_week_number_folderpath

    def save_recents_images(self) -> None:
        print(f"Saving {len(self.last_images) * 2} images")
        # We multiply the length by 2 since we save the unprocessed and the generated image

        for time_saved_image_key, saved_images_dict in self.last_images.items():
            timestamp_without_milliseconds, timestamp_milliseconds_only = str(time_saved_image_key).split(".", maxsplit=1)
            formatted_date_string = datetime.utcfromtimestamp(int(timestamp_without_milliseconds)).strftime("%Y-%m-%d_%Hh%Mm%Ss") + f"-{timestamp_milliseconds_only}millios"

            if isinstance(self.additional_text_to_use_in_filenames, str) and self.additional_text_to_use_in_filenames.replace(" ", "") != "":
                # If we have some additional text to use in the filenames, we remove the potential special chars from the text to make sure windows do not explode
                formatted_date_string = f"{''.join(char for char in self.additional_text_to_use_in_filenames if char.isalnum())}_{formatted_date_string}"

            if isinstance(saved_images_dict, dict):
                if self.KEY_UNPROCESSED in saved_images_dict.keys():
                    unprocessed_image_object = saved_images_dict[self.KEY_UNPROCESSED]
                    if isinstance(unprocessed_image_object, Image.Image):
                        unprocessed_image_filename = f"{formatted_date_string}_unprocessed.jpg"
                        unprocessed_image_object.save(os.path.join(self.get_and_make_dir_to_save_images_for_current_date(), unprocessed_image_filename))

                if self.KEY_GENERATED in saved_images_dict.keys():
                    generated_image_object = saved_images_dict[self.KEY_GENERATED]
                    if isinstance(generated_image_object, Image.Image):
                        generated_image_filename = f"{formatted_date_string}_generated.jpg"
                        generated_image_object.save(os.path.join(self.get_and_make_dir_to_save_images_for_current_date(), generated_image_filename))
