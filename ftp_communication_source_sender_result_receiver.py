import ftplib
import threading
import time
import cv2
import os
from PIL import Image

from emotions_recognition import ftp_factory

current_dir = os.path.dirname(os.path.abspath(__file__))
temp_folderpath = os.path.join(current_dir, "temp")
if not os.path.isdir(temp_folderpath):
    os.makedirs(temp_folderpath)

class ThreadUploadImageSourceFromWebcam(threading.Thread):
    def __init__(self, video_stream: cv2.VideoCapture):
        threading.Thread.__init__(self)
        self.video_stream = video_stream

        self.temp_image_to_send_filepath = os.path.join(temp_folderpath, "image_source_to_send.jpg")
        self.ftp = ftp_factory.get_ftp()

        self.count_send_source_images = 0
        self.seconds_interval_to_wait_before_sends = 0.25

    def run(self):
        while True:
            try:
                all_files_and_folders_in_current_folder = list(self.ftp.nlst())
                if "image_source_to_retrieve.jpg" not in all_files_and_folders_in_current_folder:
                    return_code, crude_frame = self.video_stream.read()

                    if crude_frame is not None:
                        frame = cv2.cvtColor(crude_frame, cv2.COLOR_BGR2RGB)
                        unprocessed_image_object = Image.fromarray(frame)
                        unprocessed_image_object.save(self.temp_image_to_send_filepath)
                        # We save the image to a file, that we will open to send to the FTP, quickly coded.

                        binary_image_file = open(self.temp_image_to_send_filepath, "rb")

                        time_start_upload = time.time()
                        self.ftp.storbinary("STOR " + "image_source_writing.jpg", binary_image_file)
                        self.ftp.rename("image_source_writing.jpg", "image_source_to_retrieve.jpg")
                        self.count_send_source_images += 1
                        print(f"Image source #{self.count_send_source_images} has been fully send and renamed in {time.time() - time_start_upload}s")
                    else:
                        raise Exception("The webcam was not correctly identified. Try changing the index of the VideoCapture of the video_stream variable.")

            except Exception as error:
                print(f"Error : {error}")
                self.ftp = ftp_factory.get_ftp()
                print(f"Upload image source ftp client has been reinitialized")

            time.sleep(0.1)

    def quit(self):
        self.ftp.quit()


class ThreadSaveGeneratedImageFromFtp(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.ftp = ftp_factory.get_ftp()

        self.temp_image_processed_writing_filepath = os.path.join(temp_folderpath, "image_processed_received_writing.jpg")
        self.temp_image_processed_complete_filepath = os.path.join(temp_folderpath, "image_generated_received_complete.jpg")

        self.count_received_generated_images = 0
        self.received_new_received_generated_image_not_yet_displayed = False
        self.received_at_least_one_generated_image = False

    def run(self):
        while True:
            if self.received_new_received_generated_image_not_yet_displayed is False:
                try:
                    all_files_and_folders_in_current_folder = list(self.ftp.nlst())
                    if "image_generated_to_retrieve.jpg" in all_files_and_folders_in_current_folder:
                        time_start_retrieve = time.time()
                        self.ftp.retrbinary("RETR " + "image_generated_to_retrieve.jpg", open(self.temp_image_processed_writing_filepath, "wb").write)
                        self.count_received_generated_images += 1
                        self.ftp.delete("image_generated_to_retrieve.jpg")
                        print(f"Retrieved generated image #{self.count_received_generated_images} and deleted the file in the ftp in {time.time() - time_start_retrieve}")

                        if os.path.isfile(self.temp_image_processed_writing_filepath):
                            need_to_update_image = False
                            if os.path.isfile(self.temp_image_processed_complete_filepath):
                                if open(self.temp_image_processed_writing_filepath, "rb").read() != open(self.temp_image_processed_complete_filepath, "rb").read():
                                    need_to_update_image = True
                                else:
                                    print(f"Received generated image #{self.count_received_generated_images} is the same as generated image #{self.count_received_generated_images - 1}.")
                            else:
                                need_to_update_image = True

                            if self.received_at_least_one_generated_image is False:
                                print(f"Since received generated image #{self.count_received_generated_images} was the first received image, it will be displayed what so ever.")
                                need_to_update_image = True
                                self.received_at_least_one_generated_image = True

                            if need_to_update_image is True:
                                if os.path.isfile(self.temp_image_processed_complete_filepath):
                                    os.remove(self.temp_image_processed_complete_filepath)

                                os.rename(self.temp_image_processed_writing_filepath, self.temp_image_processed_complete_filepath)
                                print(f"New generated image #{self.count_received_generated_images} has been received.")
                                self.received_new_received_generated_image_not_yet_displayed = True

                except Exception as error:
                    print(f"Error : {error}")
                    self.ftp = ftp_factory.get_ftp()
                    print(f"Save generated image ftp client has been reinitialized")

            time.sleep(0.1)

    def quit(self):
        self.ftp.quit()
