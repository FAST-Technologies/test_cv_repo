import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('images/coins.png')
assert img is not None, "file could not be read, check with os.path.exists()"
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
cv.imshow('gray_coins', thresh)
cv.waitKey(0)
cv.imwrite('images/gray_coins.jpg', thresh)

### Удаление шумов
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv.dilate(opening, kernel, iterations=3)
cv.imshow('sure_bg_coins', sure_bg)
cv.waitKey(0)
cv.imwrite('images/sure_bg_coins.jpg', sure_bg)

# Separated sure foreground using distance transform
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
cv.imshow('dist_transform_coins', dist_transform)
cv.waitKey(0)
cv.imwrite('images/dist_transform_coins.jpg', dist_transform)

ret, sure_fg_conn = cv.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
sure_fg_conn = np.uint8(sure_fg_conn)
cv.imshow('sure_fg_conn', sure_fg_conn)
cv.waitKey(0)
cv.imwrite('images/sure_fg_conn.jpg', sure_fg_conn)
ret, sure_fg_sep = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg_sep = np.uint8(sure_fg_sep)
cv.imshow('sure_fg_sep', sure_fg_sep)
cv.waitKey(0)
cv.imwrite('images/sure_fg_sep.jpg', sure_fg_sep)

# Connected sure foreground using stronger blurring
blurred_gray = cv.GaussianBlur(gray_img, (15, 15), 0)  # Increased kernel size for stronger blur
ret, thresh_blurred = cv.threshold(blurred_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)  # Lower threshold to connect coins
opening_blurred = cv.morphologyEx(thresh_blurred, cv.MORPH_OPEN, kernel, iterations=2)
dist_transform_blur = cv.distanceTransform(opening_blurred, cv.DIST_L2, 5)
ret, sure_fg_conn = cv.threshold(dist_transform_blur, 0.5 * dist_transform_blur.max(), 255, 0)
sure_fg_conn = np.uint8(sure_fg_conn)

# Side-by-side comparison for photo 1 (left: connected, right: separated)
photo1 = np.hstack((sure_fg_conn, sure_fg_sep))
cv.imshow('blurred_coins', photo1)
cv.waitKey(0)
cv.imwrite('images/blurred_coins.jpg', photo1)

# Нормализуем для отображения (0-255)
dist_transform_display = cv.normalize(dist_transform, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

# Вариант 2: Separated sure foreground (больший порог)
ret, sure_fg_sep = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg_sep = np.uint8(sure_fg_sep)

# Side-by-side comparison (left: distance transform, right: thresholded)
photo2 = np.hstack((dist_transform_display, sure_fg_sep))
cv.imshow('blurred_ones_coins', photo2)
cv.waitKey(0)
cv.imwrite('images/blurred_ones_coins.jpg', photo2)

# Proceed with separated sure_fg for the rest of the process
sure_fg = sure_fg_sep

unknown = cv.subtract(sure_bg, sure_fg)
cv.imshow('unknown_coins', unknown)
cv.waitKey(0)
cv.imwrite('images/unknown_coins.jpg', unknown)

ret, markers = cv.connectedComponents(sure_fg)
markers += 1
markers[unknown == 255] = 0

markers_norm = cv.normalize(markers, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
photo2 = cv.applyColorMap(markers_norm, cv.COLORMAP_JET)
cv.imshow('jet_coins', photo2)
cv.waitKey(0)
cv.imwrite('images/jet_coins.jpg', photo2)

markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
cv.imshow('final_coins', img)
cv.waitKey(0)
cv.imwrite('images/final_coins.jpg', img)