#AI project
#Applied AI Image Reconstruction Using a Genetic Algortihm

import cv2 

#global variables
qualities = []
numParentsMating = 3

#mutation percentage
mutationPercent = 2

#chromosomes per population
solPerPop = 8
M = 10 #width
N = 10 #height


#Image Manipulation
imageRequest = input("Enter your image: ")
image = cv2.imread(imageRequest)

windowName = 'image'
cv2.imshow(windowName, image)
cv2.waitKey(0)
