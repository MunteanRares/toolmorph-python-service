import cv2 as cv

def crop_image_aspect_ratio(img, width=600):
    desired_width = width
    aspect_ratio = desired_width / img.shape[1]
    desired_height = int(img.shape[0] * aspect_ratio)

    dimensions = (desired_width, desired_height)
    resized_image = cv.resize(img, dsize=dimensions, interpolation=cv.INTER_AREA)
    return resized_image