#AI project
#Applied AI Image Reconstruction Using a Genetic Algortihm

import cv2 
from random import randint, seed
import functools
import operator
import numpy as np
import itertools

#GLOBAL VARIABLES
qualities = []
numParentsMating = 3
mutationPercent = 2
#chromosomes per population
solPerPop = 8
M = 10 #width
N = 10 #height
mutation_probability = 0.1
num_generations = 200


#IMAGE MANIPULATION
#Load the target image
#imageRequest = input("Enter your image: ")
#image = cv2.imread(imageRequest)

#Display the target image
#windowName = 'image'
#cv2.imshow(windowName, image)
#cv2.waitKey(0)

#Load the image
#face = cv2.imread('face.png')
#face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
face = cv2.imread('face.png', cv2.IMREAD_GRAYSCALE)

#Converting the 2D image array to a 1D vector
def img2vector(img_arr):
  #fv = [val for sublist in img_arr for val in sublist]
  #fv = np.reshape(a = img_arr, newshape = (functools.reduce(operator.mul, img_arr.shape)))
  #return fv
  return img_arr.flatten()/255.0

#Storing face vector in a variable
faceVector = img2vector(face) #/255 #0-black, 255-white

#Converting the vector to the image
#Convert to a matrix
def convert2img(chromo, imgShape):
  imgArr = np.reshape(chromo,imgShape)
  return imgArr

#Initialize population
def initialPop(imgShape, nIndividuals=8):
  seed(1)
  #init_population = np.random.randint(0, 2, size=(nIndividuals, imgShape[0] * imgShape[1]))
  init_population = np.empty(shape = (nIndividuals, functools.reduce(operator.mul, imgShape)), dtype = np.uint8)
  #without the loop part the code runs fine 
  #for indv_num in range(nIndividuals):
    #randomly generating inital population chromosome gene values
    #init_population[indv_num, :] = randint(0, 2, M * N)
  return init_population

#fitness function - calculates the error between a target chromosome and an individual chromosome
def fitness_fun(target_chrom, indv_chrom):
  #calculates mean squared error as fitness
  #square difference between each pair and pairs elements.
  #error = sum((x -y) **2 for x, y in zip(target_chrom, indv_chrom))/ len(target_chrom)
  #return error
  #return np.mean((target_chrom - indv_chrom)**2)
  error =-1*np.sum(np.abs(indv_chrom-target_chrom))
  return error


#MUTATION
#iterations through each individual in the pop and each gene in the chromossome
def mutation(population):
  mutated_population = np.copy(population)
  for indexIndiv in range (mutated_population.shape[0]):
    for indexGene in range (mutated_population.shape[1]):
      if np.random.rand() < mutation_probability:
        mutated_population[indexIndiv, indexGene] = 1 - mutated_population[indexIndiv, indexGene]
  return mutated_population
      
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
    def crossover(parents, imgShape, nIndividuals = 8):
           
      #defining a blank array to hold the solutions after the crossover
      new_population = np.empty(shape = (nIndividuals, functools.reduce(operator.mul, imgShape)), dtype = np.uint8)

      #storing parents for the crossover operation
      new_population[0:parents.shape[0], :] = parents

      #number of offspring to be generated
      num_newly_generated = nIndividuals - parents.shape[0]

      #all permutations for the parents selected
      parents_permutations = list(itertools.permutations(iterable = np.arrange(0, parents.shape[0]), r =2))
      #selecting some parents randomly from the permutations, was random, don't know if randint will work
      selected_permutations = randint.sample(range(len(parents_permutations)), num_newly_generated)

      comb_idx = parents.shape[0]
      for comb in range(len(selected_permutations)):
        #Generating the offspring using the permutations previously selected randomly
        selected_comb_idx = selected_permutations[comb]
        selected_comb = parents_permutations[selected_comb_idx]

        #crossover by 1/50th or 2 genes between 2 parents
        cross_size = np.int32(new_population.shape[1]/50)
        new_population[comb_idx + comb, 0:cross_size] = parents[selected_comb[0], 0:cross_size]
        new_population[comb_idx + comb, cross_size:] = parents[selected_comb[1], cross_size:]
          
      return new_population
      
    #Mutation
    mutated_population = mutation(population)
      
    #replace current pop with the mutated one
    population = mutated_population
  
  return population


#Comecar o inicio da populacao
population = initialPop((M, N), solPerPop)

#Execucao do algoritmo
for generation in range(num_generations):
  population = genetic_algorithm(faceVector, population)
  #print generations
  print (f"generation {generation +1}/{num_generations} completed")
  
#Display o resultado 
for indvChromo in population:
  reconstructImg = convert2img(indvChromo, (M, N)).astype(np.uint8)
  reconstructImgResizing = cv2.resize(reconstructImg, None, fx=5,fy=5, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Reconstructed Image", reconstructImgResizing)
cv2.waitKey(0)
  
print("END")


