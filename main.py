from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

from db import *

embeddings = {}
margin = 0
img_size = [250, 312]
img = "images/obama.png"
img2 = "images/biden.jpeg"
img3 = "images/obama2.png"
img4 = "images/obama3.png"
img5 = "images/me.jpg"
img6 = "images/me.png"

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN()

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

loaded_img = Image.open(img)
loaded_img2 = Image.open(img2)
loaded_img3 = Image.open(img3)
loaded_img4 = Image.open(img4)
loaded_img4 = loaded_img4.convert('RGB')

loaded_img5 = Image.open(img5)
loaded_img5 = loaded_img5.convert('RGB')

loaded_img6 = Image.open(img6)
loaded_img6 = loaded_img6.convert('RGB')


cropped_img = mtcnn(loaded_img)
cropped_img2 = mtcnn(loaded_img2)
cropped_img3 = mtcnn(loaded_img3)
cropped_img4 = mtcnn(loaded_img4)
cropped_img5 = mtcnn(loaded_img5)
cropped_img6 = mtcnn(loaded_img6)

# print(cropped_img6)


# img_embed = resnet(cropped_img.unsqueeze(0))
img_embed2 = resnet(cropped_img2.unsqueeze(0))
img_embed3 = resnet(cropped_img3.unsqueeze(0))
img_embed4 = resnet(cropped_img4.unsqueeze(0))
img_embed5 = resnet(cropped_img5.unsqueeze(0))
img_embed6 = resnet(cropped_img6.unsqueeze(0))

img_embed = load_tensor("obama")

print((img_embed6 - img_embed).norm())

add_to_db("me", img_embed6)

# add_to_db("obama", img_embed)