import os
import sys
import cv2
import time
from PIL import Image, ImageFile

from display_window_handlers import DisplayWindowHandlers
from display_window_handlers import DisplayWindowHandlers
from image_generation import ImageGeneration
from images_saving_handlers import ImagesSavingHandlers

window = DisplayWindowHandlers()

activate_api_infos_communication = True
activate_source_sender_result_receiver = True
activate_source_receiver_result_sender = False
if activate_source_sender_result_receiver is True and activate_source_receiver_result_sender is True:
    raise Exception("The 2 modes cannot be active at the same time.")

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

style_folder_name = "clone"


class NetworkSystem:
    def __init__(self):
        self.imageGen = ImageGeneration()
        self.imagesSaving = ImagesSavingHandlers()
        self.current_dir_path = os.path.dirname(os.path.abspath(__file__))

    def start_network_loop(self):
        if activate_api_infos_communication is True:
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

        unprocessed_image_object = None
        while True:
            if activate_source_sender_result_receiver is not True:
                # We want to save the result only on the result sender and not on the result receiver (since the received results might be compressed)
                if (self.imageGen.current_selected_style_type_or_name == "random" or self.imageGen.current_selected_style_type_or_name is None
                or self.imageGen.current_selected_style_type_or_name not in self.imageGen.netG_A2B_dict_for_all_styles.keys()):
                    self.imageGen.check_random_mode_to_set_style_to_use()
                    # The random function will set the current style name to use for the models
                else:
                    self.imageGen.current_used_style_name = self.imageGen.current_selected_style_type_or_name
                    if self.imageGen.has_style_type_just_changed:
                        self.imageGen.currently_used_netG_A2B = self.imageGen.netG_A2B_dict_for_all_styles[self.imageGen.current_used_style_name]
                if self.imageGen.has_style_type_just_changed:
                    window.set_emotion_thumbnail_image(current_used_style_name=self.imageGen.current_used_style_name)
                    self.imageGen.has_style_type_just_changed = False

            if self.imagesSaving.need_to_save_pictures is True:
                self.imagesSaving.save_recents_images()
                self.imagesSaving.need_to_save_pictures = False

            need_to_update_generated_image = False
            if activate_source_receiver_result_sender is not True and activate_source_sender_result_receiver is not True:
                # If we are using the source sender result receiver, the handling and sending of the video stream is done in its own thread, not here.
                return_code, crude_frame = video_stream.read()
                if crude_frame is not None:
                    frame = cv2.cvtColor(crude_frame, cv2.COLOR_BGR2RGB)
                    unprocessed_image_object = Image.fromarray(frame)
                    processed_generated_image = self.imageGen.process_image_source_to_generated(image_source=unprocessed_image_object)
                    need_to_update_generated_image = True

            elif activate_source_receiver_result_sender is True:
                if thread_class_save_image_source_from_ftp.image_source_been_modified_and_not_yet_used or self.imageGen.has_style_type_just_changed:
                    # No matter the situation, if the style type has changed, even if  the source image has not changed, we need to update the image.
                    unprocessed_image_object = thread_class_save_image_source_from_ftp.get_image_source()
                    processed_generated_image = self.imageGen.process_image_source_to_generated(image_source=unprocessed_image_object)
                    thread_class_save_image_source_from_ftp.image_source_been_modified_and_not_yet_used = False
                    need_to_update_generated_image = True

            elif activate_source_sender_result_receiver is True:
                # If we are using the source sender result receiver, the handling and sending of the video stream is done in its own thread, not here.
                if thread_class_save_generated_image_from_ftp.received_new_received_generated_image_not_yet_displayed or self.imageGen.has_style_type_just_changed:
                    # No matter the situation, if the style type has changed, even if  the source image has not changed, we need to update the image.
                    ImageFile.LOAD_TRUNCATED_IMAGES = True
                    # Truncated images to True in order to allow partial images to still be loaded
                    processed_generated_image = Image.open(thread_class_save_generated_image_from_ftp.temp_image_processed_complete_filepath)
                    thread_class_save_generated_image_from_ftp.received_new_received_generated_image_not_yet_displayed = False
                    need_to_update_generated_image = True


            if need_to_update_generated_image is True:
                self.imagesSaving.remove_too_old_saved_unprocessed_image()
                self.imagesSaving.last_images[str(time.time())] = {
                    self.imagesSaving.KEY_UNPROCESSED: unprocessed_image_object,
                    self.imagesSaving.KEY_GENERATED: processed_generated_image
                }

                # region Display the results on the matplotlib plot
                # We remove the potential previous thumbnail image from the screen
                if plt_thumbnail_image_object is not None:
                    try:
                        plt_thumbnail_image_object.remove()
                    except Exception:
                        pass

                generated_image_plt_object = window.show_image(processed_generated_image)
                plt_thumbnail_image_object = window.put_emotion_thumbnail_on_figure(image_plt_object=generated_image_plt_object)
                # endregion

                if activate_source_receiver_result_sender is True:
                    window.plt.savefig(os.path.join(self.current_dir_path, "temp", "image_generated_to_send.jpg"))
                    thread_class_upload_generated_image.new_generated_image_to_send_has_been_created = True

                index_image += 1

            window.plt.pause(0.01)
            # Even if the displayed image has not been modified, at every step in the loop we need to call plt.pause,
            # otherwise the previously plotted image will vanish from the plot, and the window might crash.


if __name__ == "__main__":
    networkSystem = NetworkSystem()
    networkSystem.start_network_loop()


