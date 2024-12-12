import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from db import *

webcam = cv2.VideoCapture(-1)

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

comp_tensor = load_tensor("me")

recog = 1

def get_video_feed(webcam):
    if not webcam.isOpened():
        print("problem getting video feed")
        exit()

    while True:
        ret, frame = webcam.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow('frame', frame)
        cropped_img = mtcnn(frame)
        if cropped_img != None:
            img_embed = resnet(cropped_img.unsqueeze(0))
            # print(f'found a face? {None == cropped_img}')
            if (comp_tensor - img_embed).norm() <= recog:
                print("recognised me")
        if cv2.waitKey(1) == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()

get_video_feed(webcam)