import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img= cv.imread("cat/cat.png")
# Create SURF object.
# Here set Hessian Threshold to 500
surf = cv.xfeatures2d.SURF_create(500)

# Find keypoints and descriptors directly
kp, des = surf.detectAndCompute(img,None)

# Hessian threshold
surf.setHessianThreshold(500)

# Again compute keypoints and check its number.
kp, des = surf.detectAndCompute(img,None)

img2 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
cv.imshow("cat",img2)

#U-SURF, so that it won't find the orientation
surf.setUpright(True)
# Recompute the feature points and draw it
kp = surf.detect(img,None)
img3 = cv.drawKeypoints(img,kp,None,(255,0,0),4)
cv.imshow("cat",img3)

#The descriptor size and change it to 128 if it is only 64-dim.
surf.getExtended()
# So we make it to True to get 128-dim descriptors.
surf.setExtended(True)
kp, des = surf.detectAndCompute(img,None)
print( surf.descriptorSize() )

# Convert the training image to RGB
training_img = cvtest_gray = cv.cvtColor(img, cvtest_gray = cv.COLOR_BGR2RGB)

# Convert the training image to gray scale
training_gray = cvtest_gray = cv.cvtColor(training_img, cvtest_gray = cv.COLOR_RGB2GRAY)

# Create test image by adding Scale Invariance and Rotational Invariance
test_img = cvtest_gray = cv.pyrDown(training_img)
test_img = cvtest_gray = cv.pyrDown(test_img)
rows, cols = test_img.shape[:2]

rotation_matrix = cvtest_gray = cv.getRotationMatrix2D((cols/2, rows/2), 30, 1)
test_img = cvtest_gray = cv.warpAffine(test_img, rotation_matrix, (cols, rows))

test_gray = cv.cvtColor(test_img, cvtest_gray = cv.COLOR_RGB2GRAY)

cv.imshow("training image", training_img)
cv.imshow("test image", test_img)

surf = cv.xfeatures2d.SURF_create(400)

train_kp, train_descriptor = surf.detectAndCompute(training_gray, None)
test_kp, test_descriptor = surf.detectAndCompute(test_gray, None)

kp_without_size = np.copy(training_img)
kp_with_size = np.copy(training_img)

cv.drawKeypoints(training_img, train_kp, kp_without_size, color = (0, 255, 0))

cv.drawKeypoints(training_img, train_kp, kp_with_size, flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow("Train keypoints With Size",kp_with_size, cmap='gray')
cv.imshow("Train keypoints Without Size",kp_without_size, cmap='gray')

# Create a Brute Force Matcher object.
bf = cv.BFMatcher(cv.NORM_L1, crossCheck = False)
matches = bf.match(train_descriptor, test_descriptor)
matches = sorted(matches, key = lambda x : x.distance)
final_img = cv.drawMatches(training_img, train_kp, test_gray, test_kp, matches, test_gray, flags = 2)
cv.imshow("cat",final_img)

cv.waitKey(0)