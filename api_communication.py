import threading
import requests
import time
from ast import literal_eval
import live_on_webcam
from live_on_webcam import NetworkSystem

API_URL_GET_ENTIRE_CHANNEL = "https://n5p1ms9q06.execute-api.eu-west-2.amazonaws.com/env"
API_URL_GET_SAVE_PICTURES = "https://n5p1ms9q06.execute-api.eu-west-2.amazonaws.com/env/save-pictures"
API_URL_STYLE_NAME = "https://n5p1ms9q06.execute-api.eu-west-2.amazonaws.com/env/style-name"
KEY_CHANNEL_ID = "channel_id"
KEY_STYLE_NAME = "style_name"
KEY_NEED_TO_SAVE_PICTURES = "need_to_save_pictures"
KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES = "additional_text_to_use_in_filenames"
CHANNEL_ID = "Marie.Desert-X09253"

last_style_names, need_to_save_pictures_of_channel = dict(), dict()
def get_channel_infos(channel_id: str, device_id: str) -> (bool, str, bool, str):
    """
    :param channel_id:
    :param device_id:
    :return bool (has style name changed)
            str (current style name, the new one if it has been changed)
            bool (need to save pictures)
    """
    try:
        response = requests.get(f"{API_URL_GET_ENTIRE_CHANNEL}?channel_id={channel_id}" + (f"&device_id={device_id}" if device_id is not None else ""))
    except Exception as e:
        print(e.response["Error"]["Message"])
    else:
        response_content = response.content
        response_dict = None
        if isinstance(response_content, bytes):
            decoded_dict = response_content.decode("utf-8").replace("true", "True").replace("false", "False")
            response_dict = literal_eval(decoded_dict)

        if isinstance(response_dict, dict):
            # We initialize the channel_id if it is missing, otherwise we would never be able to access the
            # channel_id, when we initialize it, it will trigger the check of if the style index has been modified.
            if channel_id not in last_style_names.keys():
                last_style_names[channel_id] = None

            has_style_name_changed = False
            if KEY_STYLE_NAME in response_dict.keys():
                current_selected_style_type_or_name = response_dict[KEY_STYLE_NAME]
                # We check if the last style index has changed for our specified channel (so we can handle multiple channels at the same time)
                if current_selected_style_type_or_name != last_style_names[channel_id]:
                    # We do not forget to update the last_style_names dict since we will return
                    last_style_names[channel_id] = current_selected_style_type_or_name
                    has_style_name_changed = True

            need_to_save_pictures_bool = False
            if KEY_NEED_TO_SAVE_PICTURES in response_dict.keys():
                need_to_save_pictures_bool = response_dict[KEY_NEED_TO_SAVE_PICTURES]

                if not isinstance(need_to_save_pictures_bool, bool):
                    need_to_save_pictures_bool = False

            additional_text_to_use_in_filenames = None
            if KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES in response_dict.keys():
                additional_text_to_use_in_filenames = response_dict[KEY_ADDITIONAL_TEXT_TO_USE_IN_FILENAMES]

                if not isinstance(additional_text_to_use_in_filenames, str):
                    additional_text_to_use_in_filenames = None

            # If it has not changed, we return False and the style index, and we do not need to update the dict
            return has_style_name_changed, current_selected_style_type_or_name, need_to_save_pictures_bool, additional_text_to_use_in_filenames
        else:
            raise Exception(f"The current response was not in the form of a dict {response_dict}")

def set_need_to_save_pictures(channel_id: str, new_value: bool):
    try:
        if new_value is False:
            response = requests.delete(f"{API_URL_GET_SAVE_PICTURES}?channel_id={channel_id}")
            print(response)

        elif new_value is True:
            response = requests.post(f"{API_URL_GET_SAVE_PICTURES}?channel_id={channel_id}")
            print(response)

    except Exception as e:
        print(e.response["Error"]["Message"])


class ApiListener(threading.Thread):
    def __init__(self, thread_id, thread_name, channel_id_to_check: str, parent_networkSystem: NetworkSystem):
        threading.Thread.__init__(self)

        self.thread_id = thread_id
        self.thread_name = thread_name
        self.channel_id_to_check = channel_id_to_check
        self.parent_networkSystem = parent_networkSystem

        self.device_id = None
        if live_on_webcam.activate_source_sender_result_receiver is True:
            self.device_id = 0
        elif live_on_webcam.activate_source_receiver_result_sender is True:
            self.device_id = 1
        else:
            self.device_id = 0

    def run(self):
        while True:
            style_name_has_changed, current_selected_style_type_or_name,\
            need_to_save_pictures, additional_text_to_use_in_filenames = get_channel_infos(channel_id=self.channel_id_to_check, device_id=self.device_id)
            if style_name_has_changed and self.parent_networkSystem.imageGen is not None:
                print(f"Style name has changed and is now : {current_selected_style_type_or_name}")
                self.parent_networkSystem.imageGen.current_selected_style_type_or_name = current_selected_style_type_or_name
                self.parent_networkSystem.imageGen.has_style_type_just_changed = True

            if need_to_save_pictures:
                self.parent_networkSystem.imagesSaving.need_to_save_pictures = True
                self.parent_networkSystem.imagesSaving.additional_text_to_use_in_filenames = additional_text_to_use_in_filenames
                set_need_to_save_pictures(channel_id=self.channel_id_to_check, new_value=False)

            time.sleep(1)

def trigger_async_api_listener_loop(parent_networkSystem: NetworkSystem, channel_id: str):
    # Create new threads
    thread_api_listener = ApiListener(thread_id=1, thread_name="ApiListener-1",
                                      channel_id_to_check=CHANNEL_ID, parent_networkSystem=parent_networkSystem)
    thread_api_listener.start()


if __name__ == "__main__":
    trigger_async_api_listener_loop(channel_id=CHANNEL_ID)