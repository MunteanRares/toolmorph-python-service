import io
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from U_2NET.model.u2net import U2NET
from helpers.image_manipulation import crop_image_aspect_ratio
import torchvision.transforms as transforms

class BackgroundRemover:
    def __init__(self, image_path):
        self.image_bgr = cv.imread(image_path)
        self.image_bgr = crop_image_aspect_ratio(self.image_bgr, 800)
        self.image_rgb = cv.cvtColor(self.image_bgr, cv.COLOR_BGR2RGB)
        self.image_pil = Image.fromarray(self.image_rgb)

        self.model = U2NET()
        self.model.load_state_dict(torch.load("U_2NET/saved_models/u2net/u2net.pth"))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.515, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def remove_background(self):
        output = self.initializing_u2net_model()

        mask = ((output > 0.5).cpu().numpy() * 255).astype(np.uint8)
        binary_mask_resized = cv.resize(mask[0], (self.image_bgr.shape[1], self.image_bgr.shape[0]), interpolation=cv.INTER_AREA)

        edge_mask = self.detecting_edges(binary_mask_resized)
        edge_mask = cv.erode(edge_mask, (3, 3), iterations=1)
        grabcut_mask = self.applying_grabcut(binary_mask_resized, edge_mask)

        grabcut_mask2 = np.where((grabcut_mask == 2) | (grabcut_mask == 0), 0, 1).astype(np.uint8)
        grabcut_mask2 = cv.GaussianBlur(grabcut_mask2, (21, 21), 0, sigmaY=0)
        final_masked_image = cv.bitwise_and(self.image_bgr, self.image_bgr, mask=grabcut_mask2)

        final_image_bgra = cv.cvtColor(final_masked_image, cv.COLOR_BGR2BGRA)

        alpha_mask = (grabcut_mask2 * 255).astype(np.uint8)
        final_image_bgra[:, :, 3] = alpha_mask

        _, buffer = cv.imencode(".png", final_image_bgra)
        byte_stream = io.BytesIO(buffer)
        return byte_stream

    def initializing_u2net_model(self):
        image_tensor = self.transform(self.image_pil).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tensor)[0][0]

        return output

    def applying_grabcut(self, binary_mask, edge_mask):
        grabcut_mask = np.zeros(self.image_bgr.shape[:2], np.uint8)
        grabcut_mask[binary_mask == 255] = cv.GC_FGD
        grabcut_mask[edge_mask == 255] = cv.GC_BGD
        rect = (10, 10, self.image_bgr.shape[1] - 10, self.image_bgr.shape[0] - 10)

        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv.grabCut(self.image_bgr, grabcut_mask, rect, bgd_model, fgd_model, 10, cv.GC_INIT_WITH_MASK)

        return grabcut_mask

    @staticmethod
    def detecting_edges(mask):
        grad_x = cv.Sobel(mask, cv.CV_64F, 1, 0, ksize=5)
        grad_y = cv.Sobel(mask, cv.CV_64F, 0, 1, ksize=5)
        grad_magnitude = cv.magnitude(grad_x, grad_y)
        grad_magnitude = np.uint8(np.absolute(grad_magnitude))

        _, edge_mask = cv.threshold(grad_magnitude, 0, 255, cv.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv.erode(edge_mask, kernel, iterations=1)
        return edge_mask
