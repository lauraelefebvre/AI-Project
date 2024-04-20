#AI project
#Applied AI Image Reconstruction Using a Genetic Algortihm

#import necessary libraries
import cv2 
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.metrics import structural_similarity as ssim

#GLOBAL VARIABLES
numParentsMating = 3
chromoPerPop = 100 #chromosomes per population
M = 10 #width
N = 10 #height
mutation_probability = 1
num_generations = 500


#IMAGE MANIPULATION
#load the target image
imageRequest = input("Enter your image: ")
face = cv2.imread(imageRequest, cv2.IMREAD_GRAYSCALE)

#face = cv2.imread('face.png', cv2.IMREAD_GRAYSCALE)
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
  return imgArr.astype(np.uint8) #conversion of the pixels 

#Initialize population
def initialPop(target_image, nIndividuals=8):
  imgShape = target_image.shape
    
  #Initialize an empty population
  population = np.zeros((nIndividuals, imgShape[0], imgShape[1]), dtype=np.uint8)
    
  #Generate individuals from a normal distribution centered around the target image
  for i in range(nIndividuals):
    #balck and white
    individual = np.random.randint(0, 255, size=imgShape).astype(np.uint8)
    population[i] = individual
  return population

#fitness function - calculates the error between a target chromosome and an individual chromosome
def fitness_fun(target_chrom, indv_chrom):
  target_chrom_uint8 = (target_chrom * 255).astype(np.uint8)
  indv_chrom_uint8 = (indv_chrom * 255).astype(np.uint8)
  return cv2.PSNR(target_chrom_uint8, indv_chrom_uint8)

def calculatePopFitness(target_chrom, pop):
  qualities = np.zeros(pop.shape[0])
  for indiv_num in range (pop.shape[0]):
    indv_Chromo = pop[indiv_num].reshape(target_chrom.shape)
    qualities[indiv_num] = fitness_fun(target_chrom, indv_Chromo)
  return qualities

#SELECTION
def selection(population, qualities, num_parents):
  selected_parents = np.empty((numParentsMating, *population.shape[1:]), dtype=np.uint8)
  
  #doing the selection by ranking
  selection_probs = qualities / np.sum(qualities)
    
  #selects parents based on their probability
  selected_parents_indices = np.random.choice(np.arange(len(population)), size=num_parents, replace=False, p=selection_probs)
    
  #taking the selected parent chromossome
  selected_parents = population[selected_parents_indices]
    
  return selected_parents

#MUTATION
#iterations through each individual in the pop and each gene in the chromossome
def mutation(population):
  #create a mask to determine which genes will be mutated
  mutation_mask = np.random.rand(*population.shape) < mutation_probability

  #generate random values for mutation
  mutation_values = np.random.normal(0, 12.75, size=population.shape)

  #apply mutation to the population based on the mutation mask
  mutated_population = population + mutation_values * mutation_mask

  #clip the values to ensure they remain within the valid range of 0 to 255
  mutated_population = np.clip(mutated_population, 0, 255).astype(np.uint8)

  return mutated_population


#CROSSOVER - uniform crossover 
def crossover(parents, imgShape, nIndividuals = 8):
  new_population = np.empty(shape=(chromoPerPop, imgShape[0], imgShape[1]), dtype=np.uint8)
  
  for i in range(chromoPerPop):
    parent1 = parents[np.random.randint(parents.shape[0])]
    parent2 = parents[np.random.randint(parents.shape[0])]
    #bully = bollean(array of integers 0 and 1 whith the same shape as the image)
    bully = np.random.randint(0, 2, size=imgShape).astype(np.bool_)
    offspring = np.where(bully, parent1, parent2)
    new_population[i,:,:] = offspring
  return new_population

#selecting the best individuals in the current generation as parents for producing the offspring of the next generation
def selectMatingPool(population, fitness, num_parents):
  parents = np.empty((num_parents, population.shape[1], population.shape[2]))
  
  for parent_num in range(num_parents):
    max_fitness_idx = np.where(fitness == np.max(fitness))
    max_fitness_idx = max_fitness_idx[0][0]
    parents[parent_num, :, :] = population[max_fitness_idx, :, :]
    #updating the fitness score to a low value to prevent selecting it again
    fitness[max_fitness_idx] = -99999999999
  return parents
  
def img2vector(image):
  return image.flatten()[np.newaxis, :]
  

#GENETIC ALGORITHM FUNCTION
def genetic_algorithm(target_image, population):
  if target_image is None:
    print("Error: Image not loaded.")
    return population
  
  fig = plt.figure() #creating figure object
  ims=[] #empty list to store the images
  
  target_vector = img2vector(target_image)
  for generation in range(num_generations):
    print("Generation: ", generation)
    fitness_scores = calculatePopFitness(target_vector, population)
    parents = selectMatingPool(population, fitness_scores, num_parents)
    new_population = crossover(parents, population.shape[1:])
    new_population = mutation(new_population)
    
    #highest fitness score in current population.
    best_individual_idx = np.argmax(fitness_scores)
    new_population[0] = population[best_individual_idx]
    population = new_population

    #add the best individual of this generation to the animation
    best_individual = convert2img(population[best_individual_idx], (M, N)).astype(np.uint8)
    im = plt.imshow(best_individual, animated=True)
    ims.append([im])

  #Show the animation
  ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000, repeat=False)
  plt.show()
  
  return population

#Initiate population
population = initialPop(face, chromoPerPop)

faceVector = img2vector(face)

#Execution of the algorithm
num_parents = 4
population = genetic_algorithm(faceVector, population)

  
#Display the result
for indvChromo in population:
  reconstructImg = convert2img(indvChromo, (M, N)).astype(np.uint8)
  plt.imshow(cv2.cvtColor(reconstructImg, cv2.COLOR_BGR2RGB))
  reconstructImgResizing = cv2.resize(reconstructImg, None, fx=5,fy=5, interpolation=cv2.INTER_LINEAR)
  plt.figure(figsize=(5,5))
  
#this was to create the final image using opencv, later stopped using
# cv2.namedWindow("Reconstructed Image", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Reconstructed Image", 500,500)
# cv2.imshow("Reconstructed Image", reconstructImg)
# cv2.waitKey(0)


#TESTING
print("Size of the resized image: ", reconstructImg.shape)
print("Data type of the resized image: ", reconstructImg.dtype)

#pixel values in the resized image
print("Min pixel value: ", np.min(reconstructImg))
print("Max pixel value:", np.max(reconstructImg))


print("END")
