import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt,exp

def distance(point1,point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)

img = cv2.imread("./PandaNoise.bmp", 0)
img_clear = cv2.imread("./PandaOriginal.bmp", 0)

medium = cv2.medianBlur(img, 7)

img_c2 = np.fft.fft2(medium)
img_c3 = np.fft.fftshift(img_c2)
img_f = img_c3 * gaussianLP(50, img.shape)
img_c4 = np.fft.ifftshift(img_f)
img_c5 = np.fft.ifft2(img_c4)

plt.subplot(151), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(152), plt.imshow(np.log(1+np.abs(img_c2)), "gray"), plt.title("Spectrum")
plt.subplot(153), plt.imshow(np.log(1+np.abs(img_c3)), "gray"), plt.title("Centered Spectrum")
plt.subplot(154), plt.imshow(np.log(1+np.abs(img_c4)), "gray"), plt.title("Decentralized")
plt.subplot(155), plt.imshow(np.abs(img_c5), "gray"), plt.title("Processed Image")

filtered_img = np.abs(img_c5)
filtered_img -= filtered_img.min()
filtered_img = filtered_img*255 / filtered_img.max()
filtered_img = filtered_img.astype(np.uint8)

sharpened_image = unsharp_mask(filtered_img)

cv2.imshow('original noisy',img)
cv2.imshow('original clear',img_clear)
cv2.imshow('medium',medium)
cv2.imshow('both',filtered_img)
cv2.imshow('both + sharp',sharpened_image)

cv2.waitKey(0)

plt.show()


