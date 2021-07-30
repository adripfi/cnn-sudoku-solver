import cv2
import numpy as np
from skimage.segmentation import clear_border


class CellExtractor:
    def __init__(self, img_path, sudoku_size=(9, 9), warp_size=(800, 800), digit_size=(28, 28)):
        self.img = self._load_img(img_path)
        self.sudoku_size = sudoku_size
        self.warp_size = warp_size
        self.digit_size = digit_size
        self.warped_img = None

    @staticmethod
    def _load_img(path):
        """
        Load given image as greyscale
        """
        img = cv2.imread(path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _find_outline(self):
        """
        Detect edges of sudoku grid by searching for the biggest continuous contour within the image
        """
        # preprocess image
        blur = cv2.GaussianBlur(self.img.copy(), (7, 7), 3)

        # convert to binary using adaptive threshold
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh)

        # find contour polygons
        contours, h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # sort contours polys by their area descending and extract biggest one
        poly = max(contours, key=cv2.contourArea)

        # calculate x+y and x-y values for every point in contour polygon
        sum_pt = poly[:, 0, 0] + poly[:, 0, 1]
        diff_pt = poly[:, 0, 0] - poly[:, 0, 1]

        # find edges i.e. points with largest sum or difference of x,y values
        bot_right = poly[np.argmax(sum_pt)][0]
        top_left = poly[np.argmin(sum_pt)][0]
        top_right = poly[np.argmax(diff_pt)][0]
        bot_left = poly[np.argmin(diff_pt)][0]
        outline = np.float32([top_left, top_right, bot_left, bot_right])

        return outline

    def _warp_image(self, edges):
        """
        Transform warped sudoku grid into a rectangular one
        """
        # edges of new warped image
        new_edges = np.float32([[0, 0], [self.warp_size[0], 0], [0, self.warp_size[1]],
                                [self.warp_size[0], self.warp_size[1]]])
        # map given edges to new ones
        m = cv2.getPerspectiveTransform(edges, new_edges)

        return cv2.warpPerspective(self.img, m, self.warp_size)

    def _find_digit(self, roi):
        """
        Detect digit within given ROI by searching for the biggest contour, afterwards mask digit
        """
        # extract cell
        cell = self.warped_img[roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]]

        # convert cell to binary
        thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        # remove cell border
        thresh = clear_border(thresh)

        # if less than 3% of the binary image is filled no digit is present
        if cv2.countNonZero(thresh) / (thresh.shape[0]*thresh.shape[1]) < 0.03:
            return 0

        # find contours polygons
        contours, h = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find biggest contour by area
        poly = max(contours, key=cv2.contourArea)

        # mask digit i.e. biggest contour found
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [poly], -1, 255, -1)
        digit = cv2.bitwise_and(thresh, thresh, mask=mask)

        return cv2.resize(digit, (self.digit_size[0], self.digit_size[1]))

    def extract(self):
        """
        Main routine that finds sudoku grid, transforms it and extracts masked digits for each cell
        """
        # find outline of sudoku
        edges = self._find_outline()
        # warp image
        self.warped_img = self._warp_image(edges)

        # pixel step size between cells
        steps_x = self.warp_size[0] // self.sudoku_size[0]
        steps_y = self.warp_size[1] // self.sudoku_size[1]

        # iterate over each cell and extract digit within cell
        digits = []
        for j in range(self.sudoku_size[0]):
            for i in range(self.sudoku_size[1]):
                # cell edges
                x_start = i * steps_x
                x_end = x_start + steps_x
                y_start = j * steps_y
                y_end = y_start + steps_y
                # extract digit
                digits.append(self._find_digit([(x_start, x_end), (y_start, y_end)]))

        return digits
