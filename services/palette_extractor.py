import cv2 as cv
import numpy as np

class PaletteExtractor:
    def __init__(self, image):
        self.image = cv.imread(image)
        self.image_rgb = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)

    def extract_palette(self):
        image_small = self._crop_image_aspect_ratio(self.image_rgb, 600)
        height, width, color_amount = np.shape(image_small)
        data = np.reshape(image_small, (width * height, 3))
        data = np.float32(data)

        brightness = np.mean(data, axis=1)
        saturation = np.max(data, axis=1) - np.min(data, axis=1)
        mask = (brightness > 40) & (brightness < 220) & (saturation > 20)

        filtered_data = data[mask]
        if len(filtered_data) < 6:
            # Fallback: choose 6 random colors from original data to avoid slowdown
            if len(data) >= 6:
                idx = np.random.choice(len(data), 6, replace=False)
                filtered_data = data[idx]
            else:
                filtered_data = data

        clusters = 6
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
        flags = cv.KMEANS_PP_CENTERS
        compactness, labels, centers = cv.kmeans(filtered_data, clusters, None, criteria, 15 , flags)
        rgb_values = ['#%02x%02x%02x' % tuple(np.uint8(center)) for center in centers]

        return rgb_values

    @staticmethod
    def _crop_image_aspect_ratio(img, width):
        desired_width = width
        aspect_ratio = desired_width / img.shape[1]
        desired_height = int(img.shape[0] * aspect_ratio)

        dimensions = (desired_width, desired_height)
        resized_image = cv.resize(img, dsize=dimensions, interpolation=cv.INTER_AREA)
        return resized_image
