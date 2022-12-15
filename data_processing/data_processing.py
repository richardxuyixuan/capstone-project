from __future__ import annotations
import cv2
import numpy as np
import shutil
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from data_processing.classifier import ClassifierFC

from PIL import Image

def segmentation_img(input_path):
      '''
      this function will read the image from input_path 
      and return cropped image 
      this function is designed specifically for ads that 
      are a combination of image and text 
      '''
      img = cv2.imread(input_path)
      # set a color range for the white mask
      white_lower = np.asarray([230, 230, 230])
      white_upper = np.asarray([255, 255, 255])
      # apply the mask on image 
      mask = cv2.inRange(img, white_lower, white_upper)
      mask = cv2.bitwise_not(mask)
      # find all contours based on the bitwise mask result
      cnt, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      # select the region with largest contour
      largest_contour = max(cnt, key=lambda x:cv2.contourArea(x))
      bounding_rect = cv2.boundingRect(largest_contour)

      cropped_image = img[bounding_rect[1]: bounding_rect[1]+bounding_rect[3],
                  bounding_rect[0]:bounding_rect[0]+bounding_rect[2]]
      return cropped_image

def classifier_inference(image, resnet_weights_path, classifier_weights_path):
      '''
      this function will run inference with the pretrained classifier
      it processes one image per time 
      the input image has three channels in RGB order 
      '''
      torch.manual_seed(1000)
      # resize image to (3,224,224) and transform it to a tensor 
      transform =transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
      image = transform(image).unsqueeze(0)
      # load pretrained model
      resnet50 = models.resnet50(pretrained=True).to(torch.device("cuda"))
      resnet50.load_state_dict(torch.load(resnet_weights_path))
      classifier = ClassifierFC().to(torch.device("cuda"))
      classifier.load_state_dict(torch.load(classifier_weights_path))
      # feed the input to model 
      inputs = image.to(torch.device("cuda"))
      outputs = classifier(resnet50(inputs))
      #select index with maximum prediction score
      pred = outputs.max(1, keepdim=True)[1]
      pred = pred.cpu().squeeze().tolist()
      return pred

def img_process(input_image, output_image, resnet_weights_path, classifier_weights_path):
      '''
      input_image: str, path to input ad image
      output_image: str, a processed image will be written to output_image
            GIT will read the output_image to generate captions 
      '''
      classes = ['text','text+img','image']
      # read image in RGB channel 
      read_image = Image.open(input_image).convert('RGB') 
      # classify the type of input ad
      pred = classifier_inference(read_image, resnet_weights_path, classifier_weights_path)
      if pred == 1: # this ad is a combination of text and image
            # segment the image from the ad 
            read_image = segmentation_img(input_image)
            # write the segmented image to new path
            # later GIT will read the new path to generate caption 
            cv2.imwrite(output_image,read_image)      
      else:
            # for pure text and pure image ads, we simply 
            # copy & paste it to new path
            # later GIT will read the new path to generate caption
            shutil.copy(input_image, output_image)
      return True

if __name__ == '__main__':
      resnet_weights_path = "resnet50.pth"
      classifier_weights_path = "fc.pth"
      input_image = "1_10.png"
      output_image = "1_10.png"
      img_process(input_image, output_image, resnet_weights_path, classifier_weights_path)
      