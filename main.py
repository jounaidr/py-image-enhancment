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

def process(kx,ky,a,f):
    plt.figure(figsize=(12,8))
    plt.subplot(221)
    plt.imshow(f, cmap=plt.cm.gray)
    plt.subplot(222)
    plt.imshow(np.abs(np.log(a)), cmap=plt.cm.gray)

    #print np.shape(a)
    b=np.zeros_like(a)
    for i in range(660):
        for j in range(371):
            if not ( 330-kx<i<330+kx and (j<186-ky or j>186+ky)):
                b[j,i]=a[j,i]

    c=np.fft.ifft2(b)
    plt.subplot(223)
    plt.imshow(np.abs(np.log(b)), cmap=plt.cm.gray)
    plt.subplot(224)
    plt.imshow(np.abs(c), cmap=plt.cm.gray)

    return b


def meanSquareError(imageA, imageB):
    mseSumSq = 0
    for x in range(imageA.shape[0]):
        for y in range(imageB.shape[1]):
            mseSumSq += np.square(np.abs(imageA[x, y] - imageB[x, y]))

    mse = mseSumSq / (imageA.size * imageB.size)
    return mse

# IMAGE INPUT

img = cv2.imread("./PandaNoise.bmp", 0)  # input noisy image
img_clear = cv2.imread("./PandaOriginal.bmp", 0)  # input original clean image

# INITIAL SPACIAL DOMAIN FILTERS FOR RANDOM NOISE REMOVAL

kernel_size = 3
kernel_convolution = np.ones((kernel_size, kernel_size), np.float32)/25

img_spacial_convolution = cv2.filter2D(img, -1, kernel_convolution)  # convolution averaging
img_spacial_median = cv2.medianBlur(img, kernel_size)  # median blur filter
img_spacial_box = cv2.blur(img, (kernel_size,kernel_size))  # low pass filter
img_spacial_gaussian = cv2.GaussianBlur(img, (kernel_size,kernel_size), 0)  # gaussian filter
img_spacial_bilateral = cv2.bilateralFilter(img,kernel_size, 255, 255)  # bilateral filter

# FFT FOR FREQUENCY DOMAIN FILTERS FOR PERIODIC NOISE REDUCTION

img_fft_spectrum = np.fft.fft2(img_spacial_median)
img_fft_spectrum_centered = np.fft.fftshift(img_fft_spectrum)
img_fft_spectrum_decentered = np.fft.ifftshift(img_fft_spectrum_centered)
img_no_freq_filter = np.fft.ifft2(img_fft_spectrum_decentered)

# GAUSSIAN FILTER FREQUENCY DOMAIN

b = process(1,1,img_fft_spectrum_centered,img)

img_freq_gaussian_centered = b * gaussianLP(30, img.shape)
#img_freq_gaussian_centered = img_fft_spectrum_centered * gaussianLP(50, img.shape)
img_freq_gaussian_inverse = np.fft.ifftshift(img_freq_gaussian_centered)
img_freq_gaussian_processed = np.fft.ifft2(img_freq_gaussian_inverse)

print(meanSquareError(img_clear, img_freq_gaussian_processed))

#Plot high pass low pass stuff, do the other filters

filtered_img = np.abs(img_no_freq_filter)
filtered_img -= filtered_img.min()
filtered_img = filtered_img*255 / filtered_img.max()
filtered_img = filtered_img.astype(np.uint8)

sharpened_image = unsharp_mask(filtered_img)

# IMAGE PLOTS

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)  # initial spacial domain filters comparison

plt.subplot(161), plt.imshow(img, "gray"), plt.title("Original Image")
plt.subplot(162), plt.imshow(img_spacial_convolution, "gray"), plt.title("Convolution Averaging")
plt.subplot(163), plt.imshow(img_spacial_median, "gray"), plt.title("Median Spacial Domain Filter")
plt.subplot(164), plt.imshow(img_spacial_box, "gray"), plt.title("Box averaging Spacial Domain Filter")
plt.subplot(165), plt.imshow(img_spacial_gaussian, "gray"), plt.title("Gaussian Spacial Domain Filter")
plt.subplot(166), plt.imshow(img_spacial_bilateral, "gray"), plt.title("Bilateral Spacial Domain Filter")

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)  # fft spectrum no filters

plt.subplot(141), plt.imshow(img_spacial_median, "gray"), plt.title("Median Spacial Domain Filter")
plt.subplot(142), plt.imshow(np.log(1 + np.abs(img_fft_spectrum)), "gray"), plt.title("Spectrum")
plt.subplot(143), plt.imshow(np.log(1 + np.abs(img_fft_spectrum_centered)), "gray"), plt.title("Centered Spectrum")
plt.subplot(144), plt.imshow(np.log(1 + np.abs(img_fft_spectrum_decentered)), "gray"), plt.title("Decentralized")

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)  # gaussian frequency domain filters

plt.subplot(141), plt.imshow(np.log(1 + np.abs(img_freq_gaussian_centered)), "gray"), plt.title("TEST")
plt.subplot(142), plt.imshow(np.log(1 + np.abs(img_freq_gaussian_inverse)), "gray"), plt.title("TEST")
plt.subplot(143), plt.imshow(np.abs(img_freq_gaussian_processed), "gray"), plt.title("TEST")





plt.show()


