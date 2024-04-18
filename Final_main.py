#AI project
#Applied AI Image Reconstruction Using a Genetic Algortihm

import cv2 
from random import randint
import numpy as np


#GLOBAL VARIABLES
numParentsMating = 3
solPerPop = 100 #chromosomes per population
M = 10 #width
N = 10 #height
mutation_probability = 1
num_generations = 1001


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
  return imgArr.astype(np.uint8) #conversao dos pixels into int

#Initialize population
def initialPop(target_image, nIndividuals=8):
  imgShape = target_image.shape
    
  #Initialize an empty population
  population = np.zeros((nIndividuals, imgShape[0], imgShape[1]), dtype=np.uint8)
    
  #Generate individuals from a normal distribution centered around the target image
  for i in range(nIndividuals):
    #balck and white
    banana = np.random.randint(0, 255, size=imgShape).astype(np.uint8)
    #127 e o meio
    banana[banana <= 127] =0  #white 
    banana[banana > 127] = 0  #black
        
    individual = np.clip(target_image + banana, 0, 255)  # Ensure pixel values are within [0, 255]
    population[i] = individual
        
  return population

#fitness function - calculates the error between a target chromosome and an individual chromosome
def fitness_fun(target_chrom, indv_chrom):
  error = -np.sum(np.abs(target_chrom - indv_chrom))
  return error

def calculatePopFitness(target_chrom, pop):
  qualities = np.zeros(pop.shape[0])
  for indiv_num in range (pop.shape[0]):
    indv_Chromo = pop[indiv_num].reshape(target_chrom.shape)
    qualities[indiv_num] = fitness_fun(target_chrom, indv_Chromo)
  return qualities

#SELECTION
def selection(population, qualities, n=4):
  parents = []
  #index of individuals sorted by fitness
  index=np.argsort(qualities)[::-1]  
  best_parent_indx=index[0:n]
  for i in best_parent_indx:
    parents.append(population[i])
  parents=np.array(parents)
  return parents

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
    #random selection
    parent1 = parents[np.random.randint(parents.shape[0])]
    parent2 = parents[np.random.randint(parents.shape[0])]
    
    #random crossover
    crossover_point1 = np.random.randint(1, imgShape[0]*imgShape[1])
    crossover_point2 = np.random.randint(crossover_point1, imgShape[0]* imgShape[1])
        
    #Calculation
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
  
  
  for generation in range (num_generations):
    print("Generation: ", generation)
    
    #Selection
    fitness_scores = calculatePopFitness(target_image, population)
    parents = selection(population, fitness_scores, numParentsMating)
    
    #Crossover
    new_population = crossover(parents, target_shape)
    
    #Mutatation
    mutated_population = mutation(new_population)
    
    #replace current population with the mutated one
    population = mutated_population
  
  return population



#Comecar o inicio da populacao
population = initialPop(face, solPerPop)

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

#pixel values in the resized image
print("Min pixel value: ", np.min(reconstructImg))
print("Max pixel value:", np.max(reconstructImg))


print("END")
