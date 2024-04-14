#AI project
#Applied AI Image Reconstruction Using a Genetic Algortihm

import cv2 
from random import randint, seed
import functools
import operator
import numpy as np

#GLOBAL VARIABLES
qualities = []
numParentsMating = 3
#mutation percentage
mutationPercent = 2
#chromosomes per population
solPerPop = 8
M = 10 #width
N = 10 #height
mutation_probability = 0.1
num_generations = 100


#IMAGE MANIPULATION
#Load the target image
imageRequest = input("Enter your image: ")
image = cv2.imread(imageRequest)

#Display the target image
windowName = 'image'
cv2.imshow(windowName, image)
cv2.waitKey(0)

#Load the image
face = cv2.imread('face.png')
face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

#Converting the 2D image array to a 1D vector
def img2vector(img_arr):
    #fv = [val for sublist in img_arr for val in sublist]
    fv = np.reshape(a = img_arr, newshape = (functools.reduce(operator.mul, img_arr.shape)))
    return fv

#Storing face vector in a variable
faceVector = img2vector(face)/255 #0-black, 255-white

#Converting the vector to the image
#Convert to a matrix
def convert2img(chromo, imgShape):
  #imgArr = [chromo[i:i+M] for i in range(0,len(chromo), M)]
  imgArr = np.reshape(a = chromo, newshape = imgShape)
  return imgArr

def initialPop(imgShape, nIndividuals = 8):
  seed(1)
  init_population = np.empty(shape = (nIndividuals, functools.reduce(operator.mul, imgShape)), dtype = np.uint8)
  for indv_num in range(nIndividuals):
    #randomly generating inital population chromosome gene values
    init_population[indv_num, :] = randint(0, 2, M * N)
  return init_population

#fitness function - calculates the error between a target chromosome and an individual chromosome
def fitness_fun(target_chrom, indv_chrom):
    #calculates mean squared error as fitness
    #square difference between each pair and pairs elements.
    error = sum((x -y) **2 for x, y in zip(target_chrom, indv_chrom))/ len(target_chrom)
    return error

#GENETIC ALGORITHM FUNCTION
def genetic_algorithm(target_image, population):
    for generation in range (num_generations):
        
        #SELECTION
        fitness_scores = [fitness_fun(target_image, indv_chrom) for indv_chrom in population]
        #Parents based on the fitness scores
        parents = []
        for i in range(numParentsMating):
            #2 random indices/individuals in the population
            parent_indices = [randint(0, len(population) -1 )for i in range(2)]
            parent1 = population[parent_indices[0]]
            parent2 = population[parent_indices[1]]
            #comparisson between the scores of the 2 prents. Finds the best pair.
            if fitness_scores[parent_indices[0]] < fitness_scores[parent_indices[1]]:
                parents.append(parent1)
            else:
                parents.append(parent2)
                
        
        #CROSSOVER
        
        #MUTATION
