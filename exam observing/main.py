from imageai.Detection import VideoObjectDetection
import os
from matplotlib import pyplot as plt
execution_path = os.getcwd()


def forFrame(frame_number, output_array, output_count, returned_frame):
    labels = []
    sizes = []
    counter = 0
    for eachItem in output_count:
        counter += 1
        labels.append(eachItem + " = " + str(output_count[eachItem]))
        sizes.append(output_count[eachItem])

    return frame_number, output_array


video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "yolov3.pt"))
video_detector.loadModel()


video_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "record1.mp4"),
                                      output_file_path = os.path.join(execution_path, "video_frame"),
                                      frames_per_second = 30,
                                      per_frame_function = forFrame,
                                      minimum_percentage_probability = 30,
                                      return_detected_frame = True)

