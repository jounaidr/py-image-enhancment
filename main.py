import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt,exp
import warnings

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

def unsharp_mask(image, kernel_size=(3, 3), sigma=1.0, amount=1.0, threshold=0):
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

    b = np.full_like(a, np.mean(a))
    #b=np.zeros_like(a)
    for i in range(660):
        for j in range(371):
            if not ((66-kx<i<66+kx) or (198-kx<i<198+kx) or ((330-kx<i<330+kx) and (j<185.5-ky or j>185.5+ky)) or (462-kx<i<462+kx) or (594-kx<i<594+kx)): #goes 132
                b[j,i]=a[j,i]

    c=np.fft.ifft2(b)
    plt.subplot(223)
    plt.imshow(np.abs(np.log(b)), cmap=plt.cm.gray)
    plt.subplot(224)
    plt.imshow(np.abs(c), cmap=plt.cm.gray)

    return b

def removePeaksFromSpectrum(spectrum):
    highest = 0
    location = []
    peaks = [[35, 110, 260, 335], [65, 200, 465, 594]]

    SQ_Size = 10
    LN_Len = 60

    img_FR = spectrum.copy()

    # Find Centre
    for x in range(img_FR.shape[1]):
        for y in range(img_FR.shape[0]):
            if np.abs(img_FR[y][x]) > highest:
                highest = np.abs(img_FR[y][x])
                location = [y, x]

    centreY = location[0]
    centreX = location[1]

    peaks[0].append(centreY)
    peaks[1].append(centreX)

    average = np.abs(img_FR[0:10][0:10]).mean()

    for y in peaks[0]:
        for x in peaks[1]:
            if (x == centreX) and (y == centreY):
                continue
            else:
                for y1 in range(y - SQ_Size, y + SQ_Size):
                    for x1 in range(x - SQ_Size, x + SQ_Size):
                        img_FR[y1][x1] = 0
                for x2 in range(x - LN_Len, x + LN_Len):
                    img_FR[y][x2] = 0
                for y2 in range(y - int(LN_Len // 1.9), y + int(LN_Len // 1.9)):
                    img_FR[y2][x] = 0

    return img_FR

def convertFromFreqToGreyscale(filtered_img):
    filtered_img = np.abs(filtered_img)
    filtered_img -= filtered_img.min()
    filtered_img = filtered_img * 255 / filtered_img.max()
    filtered_img = filtered_img.astype(np.uint8)

    return filtered_img

def meanSquareError(imageA, imageB):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # suppress warning caused by imageA[y, x] - imageB[y, x]
        mseSumSq = 0
        for x in range(imageA.shape[1]):
            for y in range(imageA.shape[0]):
                mseSumSq += np.square(np.abs(imageA[y, x] - imageB[y, x]))

        mse = mseSumSq / (imageA.size)

        return mse

# IMAGE INPUT

img = cv2.imread("./PandaNoise.bmp", 0)  # input noisy image
img_clear = cv2.imread("./PandaOriginal.bmp", 0)  # input original clean image
print("Original image vs Noisy image mse: ", meanSquareError(img_clear, img))
print("")

# INITIAL SPACIAL DOMAIN FILTERS FOR RANDOM NOISE REMOVAL
print("INITIAL SPACIAL DOMAIN FILTERS FOR RANDOM NOISE REMOVAL")

kernel_size = 3
kernel_convolution = np.ones((kernel_size, kernel_size), np.float32)/25

img_spacial_convolution = cv2.filter2D(img, -1, kernel_convolution)  # convolution averaging
print("Spacial convolution averaging mse: ", meanSquareError(img_clear, img_spacial_convolution))

img_spacial_median = cv2.medianBlur(img, kernel_size)  # median blur filter
print("Spacial median blur filter mse: ", meanSquareError(img_clear, img_spacial_median))

img_spacial_box = cv2.blur(img, (kernel_size,kernel_size))  # low pass filter
print("Spacial 3x3 low pass filter mse: ", meanSquareError(img_clear, img_spacial_box))

img_spacial_gaussian = cv2.GaussianBlur(img, (kernel_size,kernel_size), 0)  # gaussian filter
print("Spacial gaussian filter mse: ", meanSquareError(img_clear, img_spacial_gaussian))

img_spacial_bilateral = cv2.bilateralFilter(img,kernel_size, 255, 255)  # bilateral filter
print("Spacial bilateral filter mse: ", meanSquareError(img_clear, img_spacial_bilateral))

print("")

# FFT FOR FREQUENCY DOMAIN FILTERS FOR PERIODIC NOISE REDUCTION

img_fft_spectrum = np.fft.fft2(img_spacial_median)
img_fft_spectrum_centered = np.fft.fftshift(img_fft_spectrum)

# GAUSSIAN FILTER IN FREQUENCY DOMAIN
print("GAUSSIAN FILTER IN FREQUENCY DOMAIN")

# LOW PASS

img_freq_lp_gaussian_centered = img_fft_spectrum_centered * gaussianLP(40, img.shape)  # apply gaussian low pass to centered spectrum with d 40 (ideal)
img_freq_lp_gaussian_inverse = np.fft.ifftshift(img_freq_lp_gaussian_centered)  # inverse shift filtered spectrum

img_freq_lp_gaussian_processed = np.fft.ifft2(img_freq_lp_gaussian_inverse)  # raw output of freq domain gaussian filter
print("Raw output of freq domain low pass gaussian filter mse: ", meanSquareError(img_clear, img_freq_lp_gaussian_processed))

img_freq_lp_gaussian_processed_sharpened = unsharp_mask(convertFromFreqToGreyscale(img_freq_lp_gaussian_processed))  # sharpened output of freq domain gaussian filter
print("Sharpened output of freq domain low pass gaussian filter mse: ", meanSquareError(img_clear, img_freq_lp_gaussian_processed_sharpened))

# HIGH PASS

img_freq_hp_gaussian_centered = img_fft_spectrum_centered * gaussianHP(40, img.shape)  # apply gaussian high pass to centered spectrum with d 40 (ideal)
img_freq_hp_gaussian_inverse = np.fft.ifftshift(img_freq_hp_gaussian_centered)  # inverse shift filtered spectrum

img_freq_hp_gaussian_processed = np.fft.ifft2(img_freq_hp_gaussian_inverse)  # raw output of freq domain gaussian filter
print("Raw output of freq domain high pass gaussian filter mse: ", meanSquareError(img_clear, img_freq_hp_gaussian_processed))

img_freq_hp_gaussian_processed_sharpened = unsharp_mask(convertFromFreqToGreyscale(img_freq_hp_gaussian_processed))  # sharpened output of freq domain gaussian filter
print("Sharpened output of freq domain high pass gaussian filter mse: ", meanSquareError(img_clear, img_freq_hp_gaussian_processed_sharpened))

print("")

b = process(1,73,img_fft_spectrum_centered,img)  # 73 is ideal 49.16447766070407

#b = removePeaksFromSpectrum(img_fft_spectrum_centered)

#img_freq_gaussian_centered = b #* gaussianLP(40, img.shape) #40 is ideal




#Plot high pass low pass stuff, do the other filters



#sharpened_image = cv2.medianBlur(convertFromFreqToGreyscale(img_freq_gaussian_processed), 3)
sharpened_image = unsharp_mask(convertFromFreqToGreyscale(img_freq_lp_gaussian_processed))


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

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)  # gaussian low pass frequency domain filters

plt.subplot(141), plt.imshow(np.log(1 + np.abs(img_freq_lp_gaussian_centered)), "gray"), plt.title("Centered Spectrum With LP Gaussian Filter Applied")
plt.subplot(142), plt.imshow(np.log(1 + np.abs(img_freq_lp_gaussian_inverse)), "gray"), plt.title("Inverse Spectrum With LP Gaussian Filter Applied")
plt.subplot(143), plt.imshow(np.abs(img_freq_lp_gaussian_processed), "gray"), plt.title("Raw Processed Image")
plt.subplot(144), plt.imshow(img_freq_lp_gaussian_processed_sharpened, "gray"), plt.title("Post Processing Sharpening")

plt.figure(figsize=(6.4*5, 4.8*5), constrained_layout=False)  # gaussian high pass frequency domain filters

plt.subplot(141), plt.imshow(np.log(1 + np.abs(img_freq_hp_gaussian_centered)), "gray"), plt.title("Centered Spectrum With HP Gaussian Filter Applied")
plt.subplot(142), plt.imshow(np.log(1 + np.abs(img_freq_hp_gaussian_inverse)), "gray"), plt.title("Inverse Spectrum With HP Gaussian Filter Applied")
plt.subplot(143), plt.imshow(np.abs(img_freq_hp_gaussian_processed), "gray"), plt.title("Raw Processed Image")
plt.subplot(144), plt.imshow(img_freq_hp_gaussian_processed_sharpened, "gray"), plt.title("Post Processing Sharpening")

print(np.square(np.subtract(img_clear,sharpened_image)).mean())
print(meanSquareError(img_clear, sharpened_image))

plt.show()


