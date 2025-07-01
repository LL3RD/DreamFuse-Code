import numpy as np
import cv2
from PIL import Image
import random


class ColorAug(object):
    def __init__(self, brightness_gamma=[0.65, 1.5]):
        self.brightness_gamma = brightness_gamma

    def brightness_aug(self, images):
        bright_0, bright_1 = self.brightness_gamma
        bright_log_0, bright_log_1 = np.log(bright_0), np.log(bright_1)
        aug_factor = np.random.uniform(bright_log_0, bright_log_1)
        if np.random.random() < 0.5:
            aug_gamma = np.exp(aug_factor)
            images = [(np.array(x) / 255.) ** aug_gamma for x in images]
        else:
            images = [np.clip((np.array(x) / 255.) + aug_factor / 2., 0, 1) for x in images]
        images = [Image.fromarray((x * 255).astype(np.uint8)) for x in images]
        return images

    def tone_aug(self, images):
        h = np.random.randint(0, 180, dtype=np.uint8)
        s = 255
        v = 255

        hsv_color = np.uint8([[[h, s, v]]])
        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[:, :, ::-1].copy()
        images = [np.array(x) for x in images]
        # if np.random.random() < 0.5:
        #     alpha = np.random.uniform(0.95, 1.0)
        #     images = [x * alpha + (1 - alpha) * rgb_color for x in images]
        #
        # else:
        #     alpha = np.random.uniform(0, 0.1)
        #     rgb_color_gamma = np.exp((rgb_color / 255. - 0.5) * alpha)
        #     images = [((x / 255.) ** rgb_color_gamma) * 255 for x in images]
        alpha = np.random.uniform(0.95, 1.0)
        images = [x * alpha + (1 - alpha) * rgb_color for x in images]
        images = [Image.fromarray(x.astype(np.uint8)) for x in images]
        return images

    def do_aug(self, images):
        p = random.random()
        if p < 0.25:
            images = self.tone_aug(images)
        elif p < 0.5:
            images = self.tone_aug(images)
            images = self.brightness_aug(images)
        elif p < 0.75:
            images = self.brightness_aug(images)
            images = self.tone_aug(images)
        else:
            images = self.brightness_aug(images)
        return images
