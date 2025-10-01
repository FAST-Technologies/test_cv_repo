import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('images/coins.png')
assert img is not None, "file could not be read, check with os.path.exists()"
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

### Удаление шумов
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv.dilate(opening, kernel, iterations=3)

# Преобразование расстояния
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)

# Нормализуем для отображения (0-255)
dist_transform_display = cv.normalize(dist_transform, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

# Separated sure foreground (больший порог)
ret, sure_fg_sep = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg_sep = np.uint8(sure_fg_sep)

# Side-by-side comparison (left: distance transform, right: thresholded)
photo1 = np.hstack((dist_transform_display, sure_fg_sep))
cv.imshow('blurred_coins', photo1)
cv.waitKey(0)
cv.imwrite('images/blurred_coins.jpg', photo1)

# Proceed with separated sure_fg for the rest of the process
sure_fg = sure_fg_sep
unknown = cv.subtract(sure_bg, sure_fg)

ret, markers = cv.connectedComponents(sure_fg)
markers += 1
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
cv.imshow('final_coins', img)
cv.waitKey(0)
cv.imwrite('images/final_coins.jpg', img)