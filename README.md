# py-image-enhancment
A myriad of image enhancement techniques in both the spatial and frequency domains (implemented using OpenCV in Python) inorder to remove
the periodic and random noise from the following image:

**Original** \
![PandaOriginal](https://raw.githubusercontent.com/jounaidr/py-image-enhancment/main/PandaOriginal.bmp)

**Noisy** \
![PandaNoise](https://raw.githubusercontent.com/jounaidr/py-image-enhancment/main/PandaNoise.bmp)

The following image is the best results achieved (visually, not MSE wise):

**'Boxes' Peaks Removal** \
![PandaBest](https://raw.githubusercontent.com/jounaidr/py-image-enhancment/main/docs/resources/PandaBest.bmp)

**Full mean squared error output of each technique used:**
```shell script
Original image vs Noisy image mse:  96.05755942171037

INITIAL SPACIAL DOMAIN FILTERS FOR RANDOM NOISE REMOVAL
Spacial convolution averaging mse:  93.94034958751939
Spacial median blur filter mse:  93.68903863432165
Spacial 3x3 low pass filter mse:  93.94034958751939
Spacial gaussian filter mse:  94.7805889079474
Spacial bilateral filter mse:  95.88277791390999

GAUSSIAN FILTER IN FREQUENCY DOMAIN
Raw output of freq domain low pass gaussian filter mse:  454.2613780574168
Sharpened output of freq domain low pass gaussian filter mse:  47.17682757494078
Raw output of freq domain high pass gaussian filter mse:  22043.168504926725
Sharpened output of freq domain high pass gaussian filter mse:  110.56922322960058

BUTTERWORTH FILTER IN FREQUENCY DOMAIN
Raw output of freq domain low pass ideal filter mse:  455.2064378861627
Sharpened output of freq domain low pass ideal filter mse:  67.00605652209426

IDEAL FILTER IN FREQUENCY DOMAIN
Raw output of freq domain low pass ideal filter mse:  491.5638751317793
Sharpened output of freq domain low pass ideal filter mse:  84.9374213836478

REMOVAL OF PEAKS THROUGH LINES
Raw output of freq domain peak line removal mse:  424.56975416294534
Sharpened output of freq domain peak line removal mse:  74.24329004329005

REMOVAL OF PEAKS THROUGH LINES WITH GAUSSIAN
Raw output of freq domain peak line removal with gaussian mse:  451.7543409656741
Sharpened output of freq domain peak line removal with gaussian mse:  39.457273544066

REMOVAL OF PEAKS THROUGH LINES WITH BUTTERWORTH
Raw output of freq domain peak line removal with butterworth mse:  443.1644291755206
Sharpened output of freq domain peak line removal with butterworth mse:  42.16281957036674

REMOVAL OF PEAKS THROUGH LINES WITH IDEAL
Raw output of freq domain peak line removal with ideal mse:  394.4991250221272
Sharpened output of freq domain peak line removal with ideal mse:  73.51460426366087

REMOVAL OF PEAKS THROUGH BOXES
Raw output of freq domain peak line removal mse:  424.9794643729004
Sharpened output of freq domain peak line removal mse:  81.46506983582455

REMOVAL OF PEAKS THROUGH BOXES WITH GAUSSIAN
Raw output of freq domain peak line removal with gaussian mse:  456.8478694953565
Sharpened output of freq domain peak line removal with gaussian mse:  54.30167442620273

REMOVAL OF PEAKS THROUGH BOXES WITH BUTTERWORTH
Raw output of freq domain peak line removal with butterworth mse:  448.89078664474636
Sharpened output of freq domain peak line removal with butterworth mse:  60.124487462223314

REMOVAL OF PEAKS THROUGH BOXES WITH IDEAL
Raw output of freq domain peak line removal with ideal mse:  398.8337513495415
Sharpened output of freq domain peak line removal with ideal mse:  75.99301641754472
```

[Full documentation (with comparison to matlab implementation)](https://github.com/jounaidr/py-image-enhancment/blob/main/docs/docs_with_matlab.pdf)
