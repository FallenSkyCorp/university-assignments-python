from imageai.Detection import ObjectDetection
import os
import cv2
execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolov3.pt"))
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image="image.jpg",
                                             output_image_path="imagenew.jpg",
                                             minimum_percentage_probability=50)

counter = 0

for eachObject in detections:
    if(eachObject["percentage_probability"] > 80):
        print("Координаты людей: ", eachObject["box_points"])
        print("--------------------------------")
        counter += 1

print("Количество людей на фото: ", counter)


def even_odd_sums(lst):
    even_sum = 0
    odd_sum = 0

    for number in lst:
        if number % 2 == 0:
            even_sum += number
        else:
            odd_sum += number
    return even_sum, odd_sum

