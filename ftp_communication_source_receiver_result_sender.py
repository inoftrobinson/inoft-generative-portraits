import time
import os
from PIL import Image
import threading

from emotions_recognition import ftp_factory

current_dir = os.path.dirname(os.path.abspath(__file__))
temp_folderpath = os.path.join(current_dir, "temp")
if not os.path.isdir(temp_folderpath):
    os.makedirs(temp_folderpath)


class ThreadSaveImageSourceFromFtp(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.temp_image_source_received_writing_filepath = os.path.join(temp_folderpath, "image_source_received_writing.jpg")
        self.temp_image_source_received_complete_filepath = os.path.join(temp_folderpath, "image_source_received_complete.jpg")

        self.ftp = ftp_factory.get_ftp(extra_filepath_to_navigate_to="images")
        self.count_received_images = 0
        self.image_source_been_modified_and_not_yet_used = False
        self.retrieved_image_at_least_once = False

    def run(self):
        while True:
            if self.image_source_been_modified_and_not_yet_used is False:
                # We make sure that the image has been used before changing it, otherwise we
                # might be writing again to the filepath while or before its being accessed.
                try:
                    all_files_and_folders_in_current_folder = list(self.ftp.nlst())
                    if "image_source_to_retrieve.jpg" in all_files_and_folders_in_current_folder:
                        time_start_retrieve = time.time()
                        self.ftp.retrbinary("RETR " + "image_source_to_retrieve.jpg", open(self.temp_image_source_received_writing_filepath, "wb").write)
                        self.ftp.delete("image_source_to_retrieve.jpg")
                        print(f"Delay to retrieve and delete the image source from the ftp : {time.time() - time_start_retrieve}s")

                        if os.path.isfile(self.temp_image_source_received_writing_filepath):
                            need_to_update_image = False
                            if os.path.isfile(self.temp_image_source_received_complete_filepath):
                                if open(self.temp_image_source_received_writing_filepath, "rb").read() != open(self.temp_image_source_received_complete_filepath, "rb").read():
                                    need_to_update_image = True
                            else:
                                need_to_update_image = True

                            if self.retrieved_image_at_least_once is False:
                                need_to_update_image = True
                                self.retrieved_image_at_least_once = True

                            if need_to_update_image is True:
                                if os.path.isfile(self.temp_image_source_received_complete_filepath):
                                    os.remove(self.temp_image_source_received_complete_filepath)

                                os.rename(self.temp_image_source_received_writing_filepath, self.temp_image_source_received_complete_filepath)

                                self.image_source_been_modified_and_not_yet_used = True
                                self.count_received_images += 1
                                print(f"The #{self.count_received_images} image source has been received.")

                except Exception as error:
                    print(f"Error : {error}")
                    self.ftp = ftp_factory.get_ftp(extra_filepath_to_navigate_to="images")
                    print(f"Save image source ftp client has been reinitialized")

            time.sleep(0.1)

    def get_image_source(self) -> Image.Image:
        if os.path.isfile(self.temp_image_source_received_complete_filepath):
            processed_generated_image = Image.open(self.temp_image_source_received_complete_filepath)
            return processed_generated_image
        else:
            return None

    def quit(self):
        self.ftp.quit()


class ThreadUploadGeneratedImage(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.temp_image_generated_to_send_filepath = os.path.join(temp_folderpath, "image_generated_to_send.jpg")
        self.ftp = ftp_factory.get_ftp(extra_filepath_to_navigate_to="images")
        self.count_send_generated_images = 0
        self.last_modified_time_of_image_generated_to_send = 0
        self.new_generated_image_to_send_has_been_created = False
        self.last_generated_image_has_completed_its_upload = True

    def inform_new_image_to_send(self):
        self.last_generated_image_has_completed_its_upload = False
        self.new_generated_image_to_send_has_been_created = True

    def run(self):
        while True:
            if self.new_generated_image_to_send_has_been_created is True:
                # We check if we need to send a new generated image, like so we do not spam the
                # network, but only send the image if it is a new one that has been generated.
                try:
                    all_files_and_folders_in_current_folder = list(self.ftp.nlst())
                    if "image_generated_to_retrieve.jpg" not in all_files_and_folders_in_current_folder:
                        if os.path.isfile(self.temp_image_generated_to_send_filepath):
                            current_file_last_modified_time = os.path.getmtime(self.temp_image_generated_to_send_filepath)
                            if current_file_last_modified_time != self.last_modified_time_of_image_generated_to_send:
                                binary_image_file = open(self.temp_image_generated_to_send_filepath, "rb")

                                time_start_upload = time.time()
                                self.ftp.storbinary("STOR " + "image_generated_writing.jpg", binary_image_file)
                                self.count_send_generated_images += 1

                                self.ftp.rename("image_generated_writing.jpg", "image_generated_to_retrieve.jpg")
                                print(f"Generated image {self.count_send_generated_images} has been fully uploaded and renamed in {time.time() - time_start_upload} seconds.")

                                self.last_modified_time_of_image_generated_to_send = current_file_last_modified_time
                                self.new_generated_image_to_send_has_been_created = False
                                self.last_generated_image_has_completed_its_upload = True
                        else:
                            raise Exception(f"No file was found at {self.temp_image_generated_to_send_filepath}")

                except Exception as error:
                    print(f"Error : {error}")
                    self.ftp = ftp_factory.get_ftp(extra_filepath_to_navigate_to="images")
                    print(f"Upload generated image ftp client has been reinitialized")

            time.sleep(0.1)

    def quit(self):
        self.ftp.quit()
