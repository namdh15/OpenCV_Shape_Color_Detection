from src.shape_detector import ShapeDetector
from src.color_detector import ColorDetector
import argparse
import imutils
import cv2
from tqdm import tqdm


def main(args):
    image = cv2.imread(args.image)
    resized = imutils.resize(image, width=300)
    
    ratio = image.shape[0] / float(resized.shape[0])
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    sd = ShapeDetector()
    cl = ColorDetector()
    
    for c in tqdm(cnts):
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)
        color = cl.detect(lab, c)
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        text = "{} {}".format(color, shape)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, text, (cX, cY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    if args.show:
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    if args.save_img:
        cv2.imwrite(args.save_path,image)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", required=False, default="./data/test.jpg",
        help="path to the input image")
    parser.add_argument("--show", action='store_true',
        help="to visualize image")
    parser.add_argument("--save_img", action='store_true',
        help="to save image")
    parser.add_argument("-o", "--save_path", required=False, default="./data/output.png",
        help="path save image")
    args = parser.parse_args()
    
    main(args)