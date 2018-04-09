"""This program stitches together images"""
import argparse
import cv2
import numpy as np

class Stitcher:
    """A class for stitching a list of images together"""
    def __init__(self, imgs, threshold=0.25, reprojection_threshold=5.0):
        self.detector = cv2.xfeatures2d.SIFT_create()
        self.imgs = [cv2.imread(img) for img in imgs]
        self.thresh = threshold
        self.reproj = reprojection_threshold
        self.min_matches = 4
        return

    def __call__(self):
        return self.stitch(self.imgs)

    def extract_features(self, img):
        """Extracts features from a single image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        return [kp, des]

    def good_features(self, img0, img1):
        features = [self.extract_features(img) for img in (img0, img1)]
        bf = cv2.BFMatcher()

        matches = bf.knnMatch(features[0][1], features[1][1], k=2)
        good = [[m] for m, n in matches if (m.distance < (self.thresh * n.distance))]
        return (features[0][0], features[1][0]), good

    def homography(self, img0, img1):
        kps, good = self.good_features(img0, img1)
        if len(good) > self.min_matches:
            ptsA = np.float32([kps[0][i[0].queryIdx].pt for i in good])
            ptsB = np.float32([kps[1][i[0].trainIdx].pt for i in good])

            H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, self.reproj)
            return H
        else:
            raise Exception("Not enough matches")


    def new_dimensions(self, canvas, img, H):
        # get canvas dimensions
        cw, ch = canvas.shape[:2]
        canvas_dims = np.float32([[0,0], [0, cw], [ch, cw], [ch, 0]]).reshape([-1, 1, 2])
        iw, ih = img.shape[:2]
        img_dims = np.float32([[0,0], [0, iw], [ih, iw], [ih, 0]]).reshape([-1, 1, 2])
        img_dims = cv2.perspectiveTransform(img_dims, H)
        resultant_dims = np.concatenate((canvas_dims, img_dims), axis=0)
        [x_min, y_min] = np.int32(resultant_dims.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(resultant_dims.max(axis=0).ravel() + 0.5)
        return [[x_min, y_min], [x_max, y_max]]

    def stitch(self, imgs):
        """Stitches a list of images together"""
        # find all of the key points and descriptors
        assert len(imgs) >= 1
        if len(imgs) == 1:
            return imgs[0]
        
        canvas = imgs[0]
        # iterate through pairs
        for idx, img in zip(range(len(imgs[:-1])), imgs[1:]):
            H = self.homography(img, canvas)
            cw, ch = canvas.shape[:2]
            new_dims = np.asarray(self.new_dimensions(canvas, img, H))
            t_offset = -new_dims[0]
            t_arr = np.array([[1, 0, t_offset[0]],
                              [0, 1, t_offset[1]],
                              [0, 0, 1]], dtype=np.int)
            new_canvas = cv2.warpPerspective(img, t_arr.dot(H),
                                             (new_dims[1, 0] - new_dims[0, 0],
                                              new_dims[1, 1] - new_dims[0, 1]))
            new_canvas[t_offset[1]:cw+t_offset[1], t_offset[0]:ch+t_offset[0]] = canvas
            canvas = new_canvas
        
        return canvas


def main():
    """The entry point to the program"""
    parser = argparse.ArgumentParser()
    parser.add_argument('imgs', type=str, nargs='+', help='The images to process')
    parser.add_argument('-t', '--thresh', type=float, default=0.25, help='The threshold for distance of correspondence')
    parser.add_argument('--display', action='store_true', help='Whether or not to display the images')
    parser.add_argument('--save', type=str, help='The name of the output image')

    args = parser.parse_args()

    stitcher = Stitcher(args.imgs, threshold=args.thresh)

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
