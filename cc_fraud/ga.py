import numpy as np
import random
from scipy.io import arff
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import time
from imblearn.over_sampling import SMOTE

start = time.time()

def get_data():
    df = pd.read_csv("creditcard_csv.csv")

    bankrupt_df = df.loc[df["Class"] == 1]
    non_bankrupt_df = df.loc[df["Class"] == 0]
    bankrupt_df = bankrupt_df.sample(n = 100, random_state=42)
    non_bankrupt_df = non_bankrupt_df.sample(n = 1000, random_state=42)
    sampled_df = pd.concat([bankrupt_df, non_bankrupt_df], axis=0, ignore_index=True)

    y = sampled_df.iloc[:, -1]

    X = sampled_df.iloc[:, :-1]

    sm = SMOTE(sampling_strategy='auto', k_neighbors=10, random_state=1)
    X_res, y_res = sm.fit_resample(X, y)

    X_res = (X_res-X_res.mean())/X_res.std()

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


class Individual:
    def __init__(self, starting_pos):
        self.genes = starting_pos
        self.fitness = -1
        self.best_model = SVC(kernel='rbf')
         
    def evaluate(self, fitness_func, X_train, X_test, y_train, y_test):
        self.fitness, self.best_model = fitness_func(X_train, X_test, y_train, y_test, self.genes)

class GeneticAlgorithm:
    def __init__(self, fitness_func,init_pos, num_individuals, bounds, max_generations, mutation_rate):
        self.fitness_func = fitness_func
        self.num_individuals = num_individuals
        self.bounds = bounds
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.best_model = SVC(kernel='rbf')
        self.X_train, self.X_test, self.y_train, self.y_test = get_data()

        #self.avg_fitness_arr = []
        self.fitness_arr = []
        
        self.population = []
        for i in range(num_individuals):
            self.population.append(Individual(init_pos[i, :]))
            
        self.best_individual = None
        self.best_fitness = -1
        
    def optimize(self):
        np.random.seed(seed=42)
        random.seed(10)
        fitness_arr = []
        best_fitness = -1
        for i in range(self.max_generations):
            # Evaluate fitness of each individual
            avg_acc = 0
            for j in range(self.num_individuals):
                self.population[j].evaluate(self.fitness_func, self.X_train, self.X_test, self.y_train, self.y_test)
                
                #avg_acc += self.population[j].fitness
                #self.fitness_arr.append(self.population[j].fitness)
                if self.population[j].fitness > best_fitness or best_fitness == -1:
                    self.best_individual = self.population[j]
                    best_fitness = self.population[j].fitness
                    self.best_model = self.population[j].best_model
            #self.avg_fitness_arr.append(avg_acc/self.num_individuals)
            
            #self.fitness_arr.append(avg_acc/self.num_individuals)
            # Create new population
            new_population = []
            
            # Elitism: always include the best individual in the new population
            new_population.append(self.best_individual)
            fitness_arr.append(best_fitness)
            
            # Roulette wheel selection
            fitness_sum = sum([indiv.fitness for indiv in self.population])
            selection_probs = [indiv.fitness/fitness_sum for indiv in self.population]
            for j in range(self.num_individuals - 1):
                new_population.append(self.population[np.random.choice(len(self.population), p=selection_probs)])
            
            # Crossover
            for j in range(0, self.num_individuals-1, 2):
                parent1 = new_population[j]
                parent2 = new_population[j+1]
                child1_new_pos = [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(len(self.bounds))]
                child2_new_pos = [random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(len(self.bounds))]
                child1 = Individual(child1_new_pos)
                child2 = Individual(child2_new_pos)
                crossover_point = random.randint(1, len(self.bounds)-1)
                child1.genes = np.concatenate((parent1.genes[:crossover_point], parent2.genes[crossover_point:]), axis=None)
                child1.genes = np.concatenate((parent2.genes[:crossover_point], parent1.genes[crossover_point:]), axis=None)
                new_population[j] = child1
                new_population[j+1] = child2
            
            # Mutation
            for j in range(1, self.num_individuals):
                for k in range(len(self.bounds)):
                    if random.uniform(0, 1) < self.mutation_rate:
                        new_population[j].genes[k] = random.uniform(self.bounds[k][0], self.bounds[k][1])
            
            self.population = new_population
        
        return (self.best_individual.genes, best_fitness, fitness_arr, self.best_model)

# Define the function to optimize (mean cross-validation score)
def svm_eval(X_train, X_test, y_train, y_test, params):

    # Create a SVM classifier object
    clf = SVC(kernel='rbf', C = params[0], gamma = params[1])

    # Train the SVM classifier
    clf.fit(X_train, y_train)

    # Predict the labels of the test set using the trained SVM classifier
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the SVM classifier on the test set
    fpr, tpr, threshold = roc_curve(y_test, y_pred)

    return accuracy_score(y_test, y_pred), clf

def main():
    np.random.seed(seed=1)
    bounds=[(0.01, 10), (0.01, 10)]
    init_pos = np.random.uniform(low=np.array([b[0] for b in bounds]), high=np.array([b[1] for b in bounds]), size=(10, len(bounds)))    
    optimizer = GeneticAlgorithm(svm_eval, init_pos, num_individuals=10, bounds=[(0.01, 10), (0.01, 10)], max_generations=100, mutation_rate=0.5)
    best_genes, best_fitness, avg_fitness_arr, svm = optimizer.optimize()

    df = pd.read_csv("creditcard_csv.csv")

    bankrupt_df = df.loc[df["Class"] == 1]
    non_bankrupt_df = df.loc[df["Class"] == 0]
    bankrupt_df = bankrupt_df.sample(n = 200, random_state=42)
    non_bankrupt_df = non_bankrupt_df.sample(n = 10000, random_state=42)
    sampled_df = pd.concat([bankrupt_df, non_bankrupt_df], axis=0, ignore_index=True)

    y = sampled_df.iloc[:, -1]

    X = sampled_df.iloc[:, :-1]

    sm = SMOTE(sampling_strategy='auto', k_neighbors=10, random_state=1)
    X_res, y_res = sm.fit_resample(X, y)

    X_res = (X_res-X_res.mean())/X_res.std()

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    # Create a SVM classifier object
    svm = SVC(kernel='rbf', C = best_genes[0], gamma = best_genes[1])

    # Train the SVM classifier
    svm.fit(X_train, y_train)

    # Predict the labels of the test set using the trained SVM classifier
    y_pred = svm.predict(X_test)

    # Compute the performance metrics

    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_test, y_pred)
    precision = tp / (tp + fp)
    sensitivity = tp / (tp + fn)
    f_measure = (2*precision*sensitivity) / (precision + sensitivity)
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    end = time.time()

    # Print the results
    print("Best Fitness: ", best_fitness)
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Sensitivity: {sensitivity:.5f}")
    print(f"Specificity: {specificity:.5f}")
    print(f"F-measure: {f_measure:.5f}")
    print(f"Confusion matrix:\n{conf_matrix}")
    print("best genes: ", best_genes)
    print("AUC: ", auc(fpr, tpr))
    print("time: ", end - start)

    a = np.asarray(avg_fitness_arr)
    np.savetxt("ga.csv", a, delimiter=",")

if __name__ == "__main__":
    main()