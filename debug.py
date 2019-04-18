import numpy as np
import cv2
import image_funcs as imf

img = cv2.imread("Images/black_board.JPG")
img = imf.scale(img, 500, 500)
cv2.imshow("small", img)
cv2.waitKey()
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
print("start kmeans")
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
print("finish kmeans")

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(img.shape)

cv2.imshow('res2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
