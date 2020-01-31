import os
import threading
import requests
import time
from ast import literal_eval

from emotions_recognition import ftp_factory
from emotions_recognition.live_on_webcam import NetworkSystem

FILENAME_CHANNEL_INFOS_TXT = "channel_infos.txt"

KEY_STYLE_NAME = "style_name"
KEY_NEED_TO_SAVE_PICTURES = "need_to_save_pictures"
KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES = "additional_text_to_use_in_filenames"
KEY_NEW_IMAGE_SOURCE_TO_RETRIEVE = "new_image_source_to_retrieve"
KEY_NEW_IMAGE_GENERATED_TO_RETRIEVE = "new_image_generated_to_retrieve"

current_dir = os.path.dirname(os.path.abspath(__file__))
temp_folderpath = os.path.join(current_dir, "temp")
if not os.path.isdir(temp_folderpath):
    os.makedirs(temp_folderpath)

# todo: complete that, and make sure that the system is stable when using a receiver, a sender, and the remote control app


class ThreadGetChannelsInfosFileFromFtp(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.ftp = ftp_factory.get_ftp()
        self.count_lines_last_retrieved_channel_infos_file = 0

        self.has_style_name_just_changed = False
        self.current_selected_style_name_or_type = None
        self.last_selected_style_name_or_type = None

        self.need_to_save_pictures = False
        self.additional_text_to_use_in_filenames = None

        self.new_image_source_to_retrieve = False
        self.new_image_generated_to_retrieve = False

    def lines_been_retrieved(self, line_string_dict: str):
        if self.count_lines_last_retrieved_channel_infos_file == 0:
            dict_channels_infos = literal_eval(line_string_dict)

            if KEY_STYLE_NAME in dict_channels_infos.keys():
                self.current_selected_style_name_or_type = dict_channels_infos[KEY_STYLE_NAME]
            if self.current_selected_style_name_or_type != self.last_selected_style_name_or_type:
                self.has_style_name_just_changed = True

            if KEY_NEED_TO_SAVE_PICTURES in dict_channels_infos.keys():
                self.need_to_save_pictures = dict_channels_infos[KEY_NEED_TO_SAVE_PICTURES]
            if KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES in dict_channels_infos.keys():
                self.additional_text_to_use_in_filenames = dict_channels_infos[KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES]

            if KEY_NEW_IMAGE_SOURCE_TO_RETRIEVE in dict_channels_infos.keys():
                self.new_image_source_to_retrieve = dict_channels_infos[KEY_NEW_IMAGE_SOURCE_TO_RETRIEVE]
            if KEY_NEW_IMAGE_GENERATED_TO_RETRIEVE in dict_channels_infos.keys():
                self.new_image_generated_to_retrieve = dict_channels_infos[KEY_NEW_IMAGE_GENERATED_TO_RETRIEVE]

        self.count_lines_last_retrieved_channel_infos_file += 1

    def run(self):
        while True:
            try:
                all_files_and_folders_in_current_folder = list(self.ftp.nlst())
                if FILENAME_CHANNEL_INFOS_TXT in all_files_and_folders_in_current_folder:
                    time_start_retrieve = time.time()

                    self.count_lines_last_retrieved_channel_infos_file = 0
                    self.ftp.retrlines(f"RETR {FILENAME_CHANNEL_INFOS_TXT}", self.lines_been_retrieved)
                    print(f"Delay to retrieve the channel infos from the ftp : {time.time() - time_start_retrieve}s")

            except Exception as error:
                print(f"Error : {error}")
                self.ftp = ftp_factory.get_ftp()
                print(f"Get channel infos from ftp ftp client has been reinitialized")

            time.sleep(0.1)
ThreadGetChannelsInfosFileFromFtp().run()


class ThreadUploadChannelInfosToFtp(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.ftp = ftp_factory.get_ftp()
        self.count_lines_last_retrieved_channel_infos_file = 0

        self.has_style_name_just_changed = False
        self.current_selected_style_name_or_type = None
        self.last_selected_style_name_or_type = None
        self.need_to_save_pictures = False
        self.additional_text_to_use_in_filenames = None

        self.finished_upload_of_last_channel_infos = False

    def lines_been_retrieved(self, line_string_dict: str):
        if self.count_lines_last_retrieved_channel_infos_file == 0:
            dict_channels_infos = literal_eval(line_string_dict)

            if KEY_STYLE_NAME in dict_channels_infos.keys():
                self.current_selected_style_name_or_type = dict_channels_infos[KEY_STYLE_NAME]
            if self.current_selected_style_name_or_type != self.last_selected_style_name_or_type:
                self.has_style_name_just_changed = True

            if KEY_NEED_TO_SAVE_PICTURES in dict_channels_infos.keys():
                self.need_to_save_pictures = dict_channels_infos[KEY_NEED_TO_SAVE_PICTURES]
            if KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES in dict_channels_infos.keys():
                self.additional_text_to_use_in_filenames = dict_channels_infos[KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES]

        self.count_lines_last_retrieved_channel_infos_file += 1

    def run(self):
        while True:
            try:
                with open(os.path.join(temp_folderpath, FILENAME_CHANNEL_INFOS_TXT), "w") as txt_file:
                    txt_file.write(str({"test": 10000}))

                with open(os.path.join(temp_folderpath, FILENAME_CHANNEL_INFOS_TXT), "rb") as txt_file:
                    time_start_upload = time.time()
                    self.ftp.storlines(f"STOR {FILENAME_CHANNEL_INFOS_TXT}", txt_file)
                    print(f"Delay to upload the channel infos to the ftp : {time.time() - time_start_upload}s")

            except Exception as error:
                print(f"Error : {error}")
                self.ftp = ftp_factory.get_ftp()
                print(f"Get channel infos from ftp ftp client has been reinitialized")

            time.sleep(0.1)
# ThreadUploadChannelInfosToFtp().run()



# def get_channel_infos(channel_id: str) -> (bool, str, bool, str):
#    # If it has not changed, we return False and the style index, and we do not need to update the dict
#    return has_style_name_changed, current_selected_style_type_or_name, need_to_save_pictures_bool, additional_text_to_use_in_filenames

class ApiListener(threading.Thread):
    def __init__(self, thread_id, thread_name, channel_id_to_check: str, parent_networkSystem: NetworkSystem):
        threading.Thread.__init__(self)

        self.thread_id = thread_id
        self.thread_name = thread_name
        self.channel_id_to_check = channel_id_to_check
        self.parent_networkSystem = parent_networkSystem

    def run(self):
        while True:
            """
            style_name_has_changed, current_selected_style_type_or_name, need_to_save_pictures, additional_text_to_use_in_filenames = get_channel_infos(channel_id=self.channel_id_to_check)
            if style_name_has_changed:
                print(f"Style name has changed and is now : {current_selected_style_type_or_name}")
                self.parent_networkSystem.current_selected_style_type_or_name = current_selected_style_type_or_name
                self.parent_networkSystem.has_style_type_just_changed = True

            if need_to_save_pictures:
                self.parent_networkSystem.need_to_save_pictures = True
                self.parent_networkSystem.additional_text_to_use_in_filenames = additional_text_to_use_in_filenames
                # set_need_to_save_pictures(channel_id=self.channel_id_to_check, new_value=False)
            """

            time.sleep(0.5)

def trigger_async_api_listener_loop(parent_networkSystem: NetworkSystem):
    # Create new threads
    thread_api_listener = ApiListener(thread_id=1, thread_name="ApiListener-1", parent_networkSystem=parent_networkSystem)
    thread_api_listener.start()


if __name__ == "__main__":
    trigger_async_api_listener_loop()