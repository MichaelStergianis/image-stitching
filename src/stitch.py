"""This program stitches together images"""
import argparse
import cv2
import numpy as np

class Stitcher:
    """A class for stitching a list of images together"""
    def __init__(self, imgs, threshold=0.4):
        self.detector = cv2.xfeatures2d.SIFT_create()
        self.imgs = [cv2.imread(img) for img in imgs]
        self.thresh = threshold
        return

    def __call__(self):
        return self.stitch(self.imgs)

    def extract_features(self, img):
        """Extracts features from a single image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        return [kp, des]

    def good_features(self, imgs):
        features = [self.extract_features(img) for img in imgs]
        bf = cv2.BFMatcher()

        matches = bf.knnMatch(features[0][1], features[1][1], k=2)
        good = [[m] for m, n in matches if (m.distance < (self.thresh * n.distance))]
        sorted(good, key=lambda x: x[0].distance, reverse=True)
        return good

    def stitch(self, imgs):
        """Stitches a list of images together"""
        # find all of the key points and descriptors
        assert len(imgs) >= 1
        if len(imgs) == 1:
            return imgs[0]
        
        good = self.good_features(imgs[:2])
        
        return


def main():
    """The entry point to the program"""
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs', type=str, nargs='+', help='The images to process')
    parser.add_argument('--display', action='store_true', help='Whether or not to display the images')
    parser.add_argument('--save', type=str, help='The name of the output image')

    args = parser.parse_args()

    stitcher = Stitcher(args.imgs)

    out = stitcher()

    # save the image
    if args.save is not None:
        cv2.imwrite(args.save, out)

    # display the image
    if args.display:
        print("Press q to quit")
        cv2.imshow('Stitched Image', out)
        while True:
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
    return


if __name__ == '__main__':
    main()
