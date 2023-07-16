import cv2
import numpy as np

# Load images
person = cv2.imread('person.jpg')
shirt = cv2.imread('shirt.jpg')

# Define points where the shirt will be placed
pts1 = np.float32([[0,0],[500,0],[0,500],[500,500]]) # These points need to be predefined based on your image
pts2 = np.float32([[220,370],[450,370],[200,480],[470,480]]) # These points need to be predefined based on your image

# Get transformation matrix
M = cv2.getPerspectiveTransform(pts1, pts2)

# Warp shirt image
warped_shirt = cv2.warpPerspective(shirt, M, (person.shape[1], person.shape[0]))

# Create mask from warped shirt image
mask = cv2.cvtColor(warped_shirt, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY_INV)

# Bitwise AND person and mask
result = cv2.bitwise_and(person, person, mask=mask)

# Add person and warped shirt images
result += warped_shirt

# Display image
cv2.imshow('Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
