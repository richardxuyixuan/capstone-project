import cv2


def augmentation_img(input_path, output_path):
      '''
      this function reads a raw image and apply blur filter and increase brightness to it
      this function is used to augment image dataset to crease more data samples 
      '''
      img = cv2.imread(input_path)
      # add blur filter 
      blur = cv2.blur(img,(3,3))
      # increase brightness to the image, value represents intensity 
      frame = increase_brightness(blur, value=20)
      if output_path:
            cv2.imwrite(output_path,frame)
      return frame

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


if __name__ == '__main__':
      read_file = "/home/skyler/codebase/year4/fall/mie429/raw_imgs/1_1.png"
      write_file = "/home/skyler/codebase/year4/fall/mie429/aug_imgs/1_1.png"
      augmentation_img(read_file, write_file)







      