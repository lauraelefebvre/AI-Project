#AI project
#Applied AI Image Reconstruction Using a Genetic Algortihm

import cv2 
from random import randint, seed
import functools
import operator
import numpy as np
import itertools

#GLOBAL VARIABLES
numParentsMating = 3
solPerPop = 8 #chromosomes per population
M = 10 #width
N = 10 #height
mutation_probability = 0.1
num_generations = 201


#IMAGE MANIPULATION
face = cv2.imread('face.png', cv2.IMREAD_GRAYSCALE)
target_shape = face.shape

#Converting the 2D image array to a 1D vector
def img2vector(img_arr):
  return img_arr.flatten()/255.0

#Storing face vector in a variable
faceVector = img2vector(face) 

#Converting the vector to the image
#Convert to a matrix
def convert2img(chromo, imgShape):
  imgArr = np.reshape(chromo,imgShape) *255.0
  return imgArr.astype(np.uint8)

#Initialize population
def initialPop(imgShape, nIndividuals=8):
  np.random.seed(1)
  init_population = np.random.randint(0, 2, size=(nIndividuals, imgShape[0] * imgShape[1]))
  init_population = init_population.reshape((nIndividuals, imgShape[0], imgShape[1]))
  return init_population

#fitness function - calculates the error between a target chromosome and an individual chromosome
def fitness_fun(target_chrom, indv_chrom):
  error = -np.mean((target_chrom - indv_chrom)**2)
  return error

def calculatePopFitness(target_chrom, pop):
  qualities = np.zeros(pop.shape[0])
  for indiv_num in range (pop.shape[0]):
    indv_Chromo = pop[indiv_num].reshape(target_chrom.shape)
    qualities[indiv_num] = fitness_fun(target_chrom, indv_Chromo)
  return qualities



#MUTATION
#iterations through each individual in the pop and each gene in the chromossome
def mutation(population):
  mutated_population = np.copy(population)
  for indexIndiv in range (mutated_population.shape[0]):
    for indexGene in range (mutated_population.shape[1]):
      if np.random.rand() < mutation_probability:
        mutated_population[indexIndiv, indexGene] = 1 - mutated_population[indexIndiv, indexGene]
  return mutated_population

#CROSSOVER
def crossover(parents, imgShape, nIndividuals = 8):
  new_population = np.empty(shape=(nIndividuals, imgShape[0], imgShape[1]), dtype=np.uint8)
  new_population[0:parents.shape[0], :, :] = parents

  # Number of offspring to be generated
  num_newly_generated = nIndividuals - parents.shape[0]
  for offspring_idx in range(num_newly_generated):
    
    parent1 = parents[np.random.randint(parents.shape[0])]
    parent2 = parents[np.random.randint(parents.shape[0])]
    
    crossover_point1 = np.random.randint(1, imgShape[0]*imgShape[1])
    crossover_point2 = np.random.randint(crossover_point1, imgShape[0]* imgShape[1])
        
    offspring = np.empty(shape=(imgShape[0], imgShape[1]), dtype=np.uint8)
    offspring_flat = offspring.flatten()
    offspring_flat[:crossover_point1] = parent1.flatten()[:crossover_point1]
    offspring_flat[crossover_point1:crossover_point2] = parent2.flatten()[crossover_point1:crossover_point2]
    offspring_flat[crossover_point2:] = parent1.flatten()[crossover_point2:]
    new_population[parents.shape[0] + offspring_idx,:,:] = offspring.reshape(imgShape)
  
  return new_population

#GENETIC ALGORITHM FUNCTION
def genetic_algorithm(target_image, population):
  if target_image is None:
    print("Erro: Imagem nao foi baixada.")
    return population
  
  #target_shape = (M,N)
  #target_image = target_image.reshape(target_shape)
  
  
  for generation in range (num_generations):
    print("Generation: ", generation)
    #SELECTION
    fitness_scores = calculatePopFitness(target_image, population)
    #Parents based on the fitness scores
    parents = []
    for i in range(numParentsMating):
      best_indices = np.argsort(fitness_scores)[-2:]
      parent_indices = np.random.choice(best_indices, size=2, replace=False)
      parents.append(population[parent_indices[0]])
      parents.append(population[parent_indices[1]])
      
    parents = np.array(parents)
    #Mutation
    new_population = crossover(parents, target_shape)
    mutated_population = mutation(new_population)
    #replace current pop with the mutated one
    population = mutated_population
    
    
  return population



#Comecar o inicio da populacao
population = initialPop((M,N), solPerPop)

#Execucao do algoritmo
population = genetic_algorithm(faceVector, population)

  
#Display o resultado 
for indvChromo in population:
  reconstructImg = convert2img(indvChromo, (M, N)).astype(np.uint8)
  reconstructImgResizing = cv2.resize(reconstructImg, None, fx=5,fy=5, interpolation=cv2.INTER_LINEAR)
  
cv2.namedWindow("Reconstructed Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Reconstructed Image", 500,500)
cv2.imshow("Reconstructed Image", reconstructImg)
cv2.waitKey(0)


#TESTING
print("Size of the resized image: ", reconstructImg.shape)
print("Data type of the resized image: ", reconstructImg.dtype)

print("Min pixel value: ", np.min(reconstructImg))
print("Max pixel value:", np.max(reconstructImg))


print("END")
