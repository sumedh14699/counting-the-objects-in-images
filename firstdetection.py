from imageai.Detection import ObjectDetection
import os
import speech_recognition as sr

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
count = 0
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "img31.jpg"), output_image_path=os.path.join(execution_path , "demoimg31.jpg"))



# get audio from the microphone
r = sr.Recognizer()

try:
    with sr.Microphone() as source:
            print("Speak:")
            audio = r.listen(source, phrase_time_limit = 5)
            print("Stop.")
            for eachObject in detections:

                if r.recognize_google(audio) in eachObject["name"]:
                    count = count+1
                    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

    print("You said " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Could not understand audio")
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))






print("objects appeared in the image are :",count)
