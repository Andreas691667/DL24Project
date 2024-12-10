import numpy as np
import cv2
import torchvision.transforms.functional as TF


class CropImgTransform:
    def __init__(self, add_pixels=0):
        self.add_pixels = add_pixels

    def __call__(self, img_):
        """
        Finds the extreme points on the image and crops the rectangular out of them
        """
        img = np.array(TF.to_pil_image(img_)) # Convert tensor to PIL image and then to numpy array
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)


        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        new_img = img[
            extTop[1] - self.add_pixels:extBot[1] + self.add_pixels,
            extLeft[0] - self.add_pixels:extRight[0] + self.add_pixels
        ].copy()

        return TF.to_tensor(new_img) # Convert numpy array back to tensor