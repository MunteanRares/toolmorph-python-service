import cv2 as cv
import numpy as np
from patsy.state import center
from scipy.spatial import distance
import matplotlib.pyplot as plt

class PaletteExtractor:
    def __init__(self, image):
        self.image = cv.imread(image)
        self.image_rgb = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)

    def extract_palette(self):
        image_small = self._crop_image_aspect_ratio(self.image_rgb, 300)
        height, width, color_amount = np.shape(image_small)
        data = np.reshape(image_small, (width * height, 3))
        data = np.float32(data)

        data = self._filter_brightness_saturation(data, 25, 225, 20)

        clusters = 24
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
        flags = cv.KMEANS_PP_CENTERS
        compactness, labels, centers = cv.kmeans(data, clusters, None, criteria, 30 , flags)

        centers = self._prioritize_distant_colors(centers, data)
        centers = self._filter_similar_color(centers, 6)
        rgb_values = ['#%02x%02x%02x' % tuple(np.uint8(center)) for center in centers]

        return rgb_values

    @staticmethod
    def _prioritize_distant_colors(centers, data):
        mean_color = np.mean(data, axis=0)
        print(mean_color)
        print(centers)
        distances = distance.cdist(centers, np.reshape(mean_color,(1, -1))).flatten()
        saturations = np.max(centers, axis=1) - np.min(centers, axis=1)
        scores = distances * saturations

        sorted_idx = np.argsort(scores)[::-1]
        top_idx = sorted_idx[:3]
        remaining_idx = [i for i in range(len(centers)) if i not in top_idx]
        combined_idx = np.concatenate((top_idx, remaining_idx))

        reordered_centers = centers[combined_idx]
        return reordered_centers

    @staticmethod
    def _filter_similar_color(centers, desired_count, threshold = 60):
        unique_centers = []
        for center in centers:
            is_unique = True
            for unique_center in unique_centers:
                if distance.euclidean(center, unique_center) <= threshold:
                    is_unique = False
                    break

            if is_unique:
                unique_centers.append(center)

            if len(unique_centers) == desired_count:
                break

        if len(unique_centers) < desired_count:
            remaining_centers = centers
            unique_centers.extend(remaining_centers[:desired_count - len(unique_centers)])

        return unique_centers

    @staticmethod
    def _filter_brightness_saturation(data, min_brightness, max_brightness, min_saturation):
        brightness = np.mean(data, axis=1)
        saturation = np.max(data, axis=1) - np.min(data, axis=1)
        mask = (brightness > min_brightness) & (brightness < max_brightness) & (saturation > min_saturation)

        filtered_data = data[mask]
        if len(filtered_data) < 6:
            if len(data) >= 6:
                idx = np.random.choice(len(data), 6, replace=False)
                filtered_data = data[idx]
            else:
                filtered_data = data

        return filtered_data


    @staticmethod
    def _crop_image_aspect_ratio(img, width = 600):
        desired_width = width
        aspect_ratio = desired_width / img.shape[1]
        desired_height = int(img.shape[0] * aspect_ratio)

        dimensions = (desired_width, desired_height)
        resized_image = cv.resize(img, dsize=dimensions, interpolation=cv.INTER_AREA)
        return resized_image
