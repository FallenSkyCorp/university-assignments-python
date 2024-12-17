from imageai.Classification import ImageClassification
from googletrans import Translator
import sys
import os

translator = Translator()
prediction = ImageClassification()

#prediction.setModelTypeAsResNet50()
#prediction.setModelPath(os.path.join(os.getcwd(), "models\\resnet50-19c8e357.pth"))

#prediction.setModelTypeAsInceptionV3()
#prediction.setModelPath(os.path.join(os.getcwd(), "models\\inception_v3_google-1a9a5a14.pth"))

#prediction.setModelTypeAsMobileNetV2()
#prediction.setModelPath(os.path.join(os.getcwd(), "models\\mobilenet_v2-b0353104.pth"))

prediction.setModelTypeAsDenseNet121()
prediction.setModelPath(os.path.join(os.getcwd(), "models\\densenet121-a639ec97.pth"))

prediction.loadModel()

folder_path = os.getcwd()
folder_path += os.sep + 'img' + os.sep

if(not os.path.exists(folder_path) or os.listdir(folder_path) == 0):
	print('Нет данных для анализа')
	sys.exit(0)

fileList = os.listdir(folder_path)

counter = 0

for file in fileList:
	if os.path.isfile(folder_path + file) and file.endswith('.JPG'):
		predictions, probabilities = prediction.classifyImage(os.path.join(folder_path, file), result_count=1)
		str_translated = translator.translate(f'{predictions[counter]}', dest='ru')
		print(file, ' - на английском: ', predictions[counter], '. На русском: ', str_translated.text)