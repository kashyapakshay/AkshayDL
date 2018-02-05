import numpy as np
import matplotlib.pyplot as plt
from random import random, randint, sample

from keras.models import load_model
from keras.datasets import mnist

class ImageEvolve:
    def __init__(self, fitness_func):
        self.fitness_func = fitness_func

    def individual(self, size=(28, 28)):
        '''
        Generate random bitmap image. Default size 28x28.
        '''

        return np.random.randint(256, size=size)

    def population(self, N):
        '''
        Generate a population of individuals,
        '''

        return np.array([self.individual() for count in xrange(N)])

    def fitness(self, individual, target):
        '''
        Calculate the fitness of an individual.
        '''

        return self.fitness_func(individual, target)

    def grade(self, population, target):
        '''
        Get mean fitness of the entire population.
        '''

        return np.mean(map(lambda individual: self.fitness(individual, target), population))

    def mutate(self, image):
        mutationProbability = 0.1
        mutated = np.copy(image)

        for i in xrange(image.shape[0]):
            for j in xrange(image.shape[1]):
                if random() > mutationProbability:
                    mutated[i, j] = randint(0, 255)

        return mutated


    def evolve(self, population, target, retain=0.1, random_select=0.1, mutate=0.4):
        '''
        Evolve the population.
        '''

        graded = [(self.fitness(individual, target), individual) for individual in population]
        graded.sort(key=lambda x: x[0])
        graded = [x[1] for x in graded]
        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]

        # randomly add other individuals to promote genetic diversity
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)

        # mutate some individuals
        for individual in parents:
            if mutate > random():
                pos_to_mutate = randint(0, individual.shape[0] - 1)
                individual[pos_to_mutate,] = randint(0, 255)
                pos_to_mutate = randint(0, individual.shape[1] - 1)
                individual[range(28),pos_to_mutate] = randint(0, 255)

                # individual[pos_to_mutate,], individual[pos_to_swap] = individual[pos_to_swap,], individual[pos_to_mutate]
                # individual[sample(range(28), 14), sample(range(28), 14)] = randint(0, 255)
                # individual = self.mutate(individual)

        # crossover parents to create children
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []
        while len(children) < desired_length:
            male = randint(0, parents_length - 1)
            female = randint(0, parents_length - 1)

            if male != female:
                male = parents[male]
                female = parents[female]
                half = len(male) / 2
                child = np.concatenate((male[:half,], female[half:,]), axis=0)
                children.append(child)

        parents.extend(children)
        return parents

def test():
    fitness_func = lambda individual, target: np.sum(individual)
    ev = ImageEvolve(fitness_func)
    population = ev.population(10)
    grade = ev.grade(population, None)

    print grade

def main():
    model = load_model('mnist_model_batch-1-stupid.h5')

    def fitness_func(individual, target):
        scores = model.predict(np.array([individual]))[0]
        scores -= np.max(scores) # normalize to avoid numerical instability

        # Softmax loss function
        actual_score = scores[target]
        loss = -actual_score + np.log(np.sum(np.exp(scores)))

        prediction = np.argmax(scores)
        if loss == 0:
            loss = 0.001

        fit = 1 / loss
        return fit

    ev = ImageEvolve(fitness_func)
    epochs = 500
    predictionComparison = []
    fooledSucess = 0

    for x in xrange(10):
        print 'FOOLING FOR MNIST IMAGE: ', x, '\n'

        target = x
        population = ev.population(50)
        score = ev.grade(population, target)
        scGrowth = []

        for i in range(epochs):
            evolved_population = ev.evolve(population, target)
            score_new = ev.grade(evolved_population, target)

            print '[Epoch] %d | [Score] %f | [Best Score] %f' % (i, score_new, score)

            if score_new > score:
                score = score_new
                population = evolved_population

            scGrowth.append(score)

        print '\nBest score: ', score

        graded = [(ev.fitness(individual, target), individual) for individual in population]
        graded.sort(key=lambda x: x[0])
        best_individual = graded[-1][1]

        # plt.imshow(best_individual)
        plt.imsave('pics/' + str(target) + '.jpg', best_individual)
        # plt.show()

        plt.plot(range(epochs), scGrowth)
        plt.ylabel('Best Fitness Score Growth')
        plt.savefig('pics/' + str(target) + '-sg.jpg')
        plt.clf()
        
        scores = model.predict(np.array([best_individual]))
        prediction = np.argmax(scores[0])
        predictionComparison.append((x, prediction, score))

        print 'Predicted [', prediction, '] for Target [', target, ']\n'

        if prediction == x:
            fooledSucess += 1

    print '\nSuccessfully Fooled ', fooledSucess,'/10 times\n'
    print 'Actual | Predicted | Score'
    print '\n'.join([str(comp) for comp in predictionComparison])

if __name__ == '__main__':
    main()
