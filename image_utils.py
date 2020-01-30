from torchvision.utils import save_image

from emotions_recognition.live_on_webcam import NetworkSystem

def save_image_tensor_to_image_file(count_generated_images: int, image_tensor: NetworkSystem.Tensor):
    # This function is not called save_image, since save_image is an util function of torchvision
    save_image(image_tensor, f"output/B/{count_generated_images}.png")

