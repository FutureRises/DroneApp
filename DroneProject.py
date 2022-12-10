# written by James Johnson for FutureRises Analytics

from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
import djitellopy


def forFrame(frame_number, output_array, output_count):
    print("For Frame ", frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("---------------------End of Frame-------------------")


# instantiating the classes from ImageAI
PicRecognizer = ObjectDetection()
VidRecognizer = VideoObjectDetection()

# define the paths
path_model = "./Models/yolo-tiny.h5"
path_input = "./Input/attachment.jpg"
path_output = "./Output/newimage.jpg"
VidPath_output = "./Output/newVideo"
VidPath_input = "./Input/Fishermans.mp4"

# using TinyYOLOv3
PicRecognizer.setModelTypeAsTinyYOLOv3()
VidRecognizer.setModelTypeAsTinyYOLOv3()

# setting the path the pre-trained model then load
PicRecognizer.setModelPath(path_model)
PicRecognizer.loadModel()
VidRecognizer.setModelPath(path_model)
PicRecognizer.loadModel()

# detect objects from image
recognition = PicRecognizer.detectObjectsFromImage(input_image=path_input, output_image_path=path_output)
VidRecognition = VidRecognizer.detectObjectsFromVideo(input_file_path=VidPath_input,
                                                      output_file_path=VidPath_output, frames_per_second=29.97,
                                                      minimum_percentage_probability=30)

# iterate through the items found in image
for eachItem in recognition:
    print(eachItem["name"], ":", eachItem["percentage_probability"])

print("And Now the Video: ")
print(VidRecognition)
