# Import necessary libraries
import cv2 
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.metrics import structural_similarity as ssim


# Set global variables
numParentsMating = 3
solPerPop = 100 #chromosomes per population
M = 10 #width
N = 10 #height
mutation_probability = 1
num_generations = 500

# Load the image
face = cv2.imread('face.png', cv2.IMREAD_GRAYSCALE)
target_shape = face.shape

# Convert the 2D image array to a 1D vector
def img2vector(img_arr):
    return img_arr.flatten()/255.0

# Convert the vector to the image
def convert2img(chromo, imgShape):
    imgArr = np.reshape(chromo,imgShape) *255.0
    return imgArr.astype(np.uint8) #conversao dos pixels into int

def initialPop(target_image, nIndividuals=8):
    imgShape = target_image.shape
    population = np.zeros((nIndividuals, imgShape[0], imgShape[1]), dtype=np.uint8)
    for i in range(nIndividuals):
        individual = np.random.randint(0, 256, size=imgShape).astype(np.uint8)
        population[i] = individual
    return population

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

def selection(population, qualities, numParentsMating):
    # Tournament selection
    selected_parents = np.empty((numParentsMating, *population.shape[1:]), dtype=np.uint8)
    for i in range(numParentsMating):
        tournament = np.random.choice(np.arange(population.shape[0]), size=3)
        best_individual = tournament[np.argmin(qualities[tournament])]
        selected_parents[i] = population[best_individual]
    return selected_parents

def crossover(parents, imgShape):
    # Uniform crossover
    new_population = np.empty(shape=(solPerPop, imgShape[0], imgShape[1]), dtype=np.uint8)
    for i in range(solPerPop):
        parent1 = parents[np.random.randint(parents.shape[0])]
        parent2 = parents[np.random.randint(parents.shape[0])]
        mask = np.random.randint(0, 2, size=imgShape).astype(np.bool_)
        offspring = np.where(mask, parent1, parent2)
        new_population[i,:,:] = offspring
    return new_population
  
def mutation(population):
    # Gaussian mutation
    mutated_population = np.copy(population).astype(np.float64)
    mutation_mask = np.random.rand(*population.shape) < mutation_probability
    mutated_population[mutation_mask] += np.random.normal(0, 12.75, size=mutated_population[mutation_mask].shape)
    mutated_population = np.clip(mutated_population, 0, 255).astype(np.uint8)
    return mutated_population

def selectMatingPool(population, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, population.shape[1], population.shape[2]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :, :] = population[max_fitness_idx, :, :]
        fitness[max_fitness_idx] = -99999999999
    return parents
  
def img2vector(image):
    return image.flatten()[np.newaxis, :]
  
def genetic_algorithm(target_image, population):
    fig = plt.figure()
    ims = []

    target_vector = img2vector(target_image)
    for generation in range(num_generations):
        print("Generation: ", generation)
        fitness_scores = calculatePopFitness(target_vector, population)
        parents = selectMatingPool(population, fitness_scores, num_parents)
        new_population = crossover(parents, population.shape[1:])
        new_population = mutation(new_population)
        # Elitism: keep the best individual from the current population
        best_individual_idx = np.argmax(fitness_scores)
        new_population[0] = population[best_individual_idx]
        population = new_population

        # Add the best individual of this generation to the animation
        best_individual = convert2img(population[best_individual_idx], (M, N)).astype(np.uint8)
        im = plt.imshow(best_individual, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    plt.show()

    return population

# Start the initial population
population = initialPop(face, solPerPop)

# Convert the face image to a vector
faceVector = img2vector(face)

# Execute the algorithm
num_parents = 4
population = genetic_algorithm(faceVector, population)

# Display the result 
for indvChromo in population:
    reconstructImg = convert2img(indvChromo, (M, N)).astype(np.uint8)
    reconstructImgResizing = cv2.resize(reconstructImg, None, fx=5,fy=5, interpolation=cv2.INTER_LINEAR)
    plt.figure(figsize=(5, 5))  # Set the figure size
    plt.imshow(cv2.cvtColor(reconstructImgResizing, cv2.COLOR_BGR2RGB))  # Convert color space from BGR to RGB

# Testing
print("Size of the resized image: ", reconstructImg.shape)
print("Data type of the resized image: ", reconstructImg.dtype)
print("Min pixel value: ", np.min(reconstructImg))
print("Max pixel value:", np.max(reconstructImg))
print("END")