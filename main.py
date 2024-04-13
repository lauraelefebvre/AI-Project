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

face = cv2.imread('face.png')
face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

def img2vector(img_arr):
  fv = np.reshape(a = img_arr, newshape = (functools.reduce(operator.mul img_arr.shape)))
  return fv

#storing face vector in a variable
faceVector = img2vector(face)/255

#convert to a matrix
def convert2img(chromo, imgShape):
  imgArr = mp.reshape(a = chromo, newshape = imgShape)
  return imgArr

def initialPop(imgShape, nIndividuals = 8):
  seed(1)
  init_population = np.empty(shape = (nIndividuals, functools.reduce(operator.mul, imgShape)), dtype = np.uint8)
  for indv_num in range(nIndividuals):
    #randomly generating inital population chromosome gene values
    init_population[indv_num, :] = randint(0, 2, M * N)
  return init_population

#fitness function
def fitness_fun(target_chrom, indv_chrom):
  
  return error