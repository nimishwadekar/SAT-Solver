from CNF_Creator import *
import random
import time
import bisect
import copy
import math

SYMBOLS = 50

REQUIRED_FITNESS = 100
TIMEOUT = 45

Sentence = []
Clauses = 0

# Returns fitness value of a state.
def Fitness(state):
    wt = 0
    for clause in Sentence:
        for variable in clause:
            if (variable > 0 and state[variable - 1]) or (variable < 0 and not state[-variable - 1]):
                wt += 1
                break
    return 100 * wt / Clauses


# Represents a state.
class State:
    def __init__(self, value):
        self.value= value
        self.fitness = Fitness(value)
    
    def __str__(self):
        return self.value.__str__()


# returns a random state.
def GenerateRandomState():
    state = [False] * SYMBOLS
    for i in range(SYMBOLS):
        if random.randint(0, 1) == 1:
            state[i] = True
    return State(state)


# Returns a cumulative list of fitness values of a population.
def GetCumulativeWeights(population):
    weights = []
    last = 0
    for state in population:
        last = last + state.fitness
        weights.append(last)
    return weights


# Chooses and returns 2 parents from the population.
def ChooseParents(population, weights):
    limit = weights[-1]

    # Selection chance proportional to fitness.
    parentIndex1 = bisect.bisect_left(weights, random.random() * limit)
    # Other parent selected is on the opposite side of parent 1, treating population as a circular list.
    parentIndex2 = (parentIndex1 + len(population) // 2) % len(population)

    return population[parentIndex1], population[parentIndex2]


# Performs crossing over of the provided parents.
def Reproduce(parents):
    parent1, parent2 = random.randint(0, SYMBOLS - 1), random.randint(0, SYMBOLS - 1)
    parent1, parent2 = min(parent1, parent2), max(parent1, parent2)

    # 2 crossover points.
    child1 = State(parents[0].value[ : parent1] + parents[1].value[parent1 : parent2] + parents[0].value[parent2 : ])
    child2 = State(parents[1].value[ : parent1] + parents[0].value[parent1 : parent2] + parents[1].value[parent2 : ])

    # Return the better child.
    return child1 if child1.fitness > child2.fitness else child2


# Performs mutation on the provided child.
def Mutate(child, stagnation):
    childValue = copy.copy(child.value)

    # Sigmoid function.
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # Denominator varies with level of stagnation.
    stagDenom = math.pow((stagnation // 50) + 1, 1.5) * 50
    # Sigmoid function used to get a value between 0 and 1.
    chance = sigmoid(stagnation * 8 / stagDenom - 4)

    # Each boolean in child has a chance for mutation.
    for pos in range(SYMBOLS):
        if random.random() < chance:
            childValue[pos] = not childValue[pos]
    mutant = State(childValue)

    # Mutate only if beneficial.
    return mutant if mutant.fitness >= child.fitness else child


# Return value: Best state, Time taken, number of generations to find best state
def GeneticAlgorithm(population):
    bestState = None # The best state found by this algorithm.
    stagnation = 0 # Variable to keep track of stagnation.
    weights = None # List of cumulative fitness values (weights) of the population.
    populationSize = len(population)
    prevBestFitness = -1 # Best fitness value found in the previous generation.

    generations = 0
    elapsedTime = 0
    startTime = time.monotonic()
    while elapsedTime < TIMEOUT:
        generations = generations + 1
        nextGeneration = []
        weights = GetCumulativeWeights(population)
        currentBestFitness, currentBestState = -1, None

        for i in range(populationSize):
            # Choose parents.
            parents = ChooseParents(population, weights)

            # Reproduce and possibly mutate.
            child = Reproduce(parents)
            child = Mutate(child, stagnation)

            # Child is chosen only if it surpasses its parents, ensuring no decline in fitness.
            child = max([child, parents[0], parents[1]], key = lambda state: state.fitness)

            # Return if satisfactory fitness found.
            if(child.fitness >= REQUIRED_FITNESS):
                return (child, time.monotonic() - startTime, generations)

            # Update this generation's best state.
            if currentBestFitness < child.fitness:
                currentBestFitness = child.fitness
                currentBestState = child

            # Add child to next generation population.
            nextGeneration.append(child)

        population = nextGeneration

        # Check for stagnation. Decline also considered stagnation.
        if currentBestFitness <= prevBestFitness:
            stagnation += 1
        else:
            stagnation = 0
        prevBestFitness = currentBestFitness
        
        # Update overall best state found.
        bestState = currentBestState if bestState is None or bestState.fitness < currentBestFitness else bestState

        elapsedTime = time.monotonic() - startTime
    
    return (bestState, elapsedTime, generations)


# Generates a random populaiton of size 'size'.
def GeneratePopulation(size):
    return [GenerateRandomState() for i in range(size)]


def main():
    cnfC = CNF_Creator(n = SYMBOLS)

    global Sentence
    Sentence = cnfC.ReadCNFfromCSVfile()

    global Clauses
    Clauses = len(Sentence)

    population = GeneratePopulation(100)
    bestModel, runtime, generations = GeneticAlgorithm(population)

    # Get numeric representation of list[bool] state.
    model = [(i + 1) * (1 if val else -1) for i, val in enumerate(bestModel.value)]

    print('\n\n')
    print('Roll No : 2019A7PS1004G')
    print('Number of clauses in CSV file :', Clauses)
    print('Best model :', model)
    print(f'Fitness value of best model : {"%.2f" % bestModel.fitness}%')
    print('Time taken : ', "%.2f" % runtime, 'seconds')
    print('\n\n')


if __name__=='__main__':
    main()