from src.shape_detector import ShapeDetector
import argparse
import imutils
import cv2
from tqdm import tqdm

def main(args):
    image = cv2.imread(args.image)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()
    for c in tqdm(cnts):
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2)
        
    if args.show:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    if args.save_img:
        cv2.imwrite(args.save_path,image)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=True, default="./data/test.jpg",
        help="path to the input image")
    parser.add_argument("--show", action='store_true',
        help="to visualize image")
    parser.add_argument("--save_img", action='store_true',
        help="to save image")
    parser.add_argument("-o", "--save_path", required=False, default="output.png",
        help="path save image")
    args = parser.parse_args()
    
    main(args)