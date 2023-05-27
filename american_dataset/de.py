import numpy as np
import random
from sklearn.model_selection import KFold
from scipy.io import arff
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, cross_val_predict
from imblearn.over_sampling import SMOTE
import time

start = time.time()

def get_data():
    df = pd.read_csv("american_bankruptcy_dataset.csv")

    df.loc[:, 'status_label'] = df.loc[:, 'status_label'].eq('failed').mul(1)
    df = df.drop(['year', 'company_name'], axis=1)

    bankrupt_df = df.loc[df["status_label"] == 1]
    non_bankrupt_df = df.loc[df["status_label"] == 0]
    bankrupt_df = bankrupt_df.sample(n = 100, random_state=42)
    non_bankrupt_df = non_bankrupt_df.sample(n = 1000, random_state=42)
    sampled_df = pd.concat([bankrupt_df, non_bankrupt_df], axis=0, ignore_index=True)

    y = sampled_df.iloc[:, 0]

    X = sampled_df.iloc[:, 1:]

    sm = SMOTE(sampling_strategy='auto', k_neighbors=10, random_state=1)
    X_res, y_res = sm.fit_resample(X, y)

    X_res = (X_res-X_res.mean())/X_res.std()

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test

def differential_evolution(obj_func, bounds, popsize=10, mutation=0.5, recombination=0.5, maxiter=100):
    """
    Differential evolution optimizer.

    Parameters:
        obj_func (function): Objective function to optimize.
        bounds (list of tuples): Bounds for each parameter. Example: [(lower_1, upper_1), (lower_2, upper_2), ...].
        popsize (int): Population size.
        mutation (float): Mutation constant.
        recombination (float): Recombination constant.
        maxiter (int): Maximum number of iterations.

    Returns:
        best_solution (array): Best solution found.
        best_fitness (float): Fitness of best solution.
    """
    # Initialize population
    fitness_arr = []

    X_train, X_test, y_train, y_test = get_data()
    # Initialize particles randomly
    np.random.seed(seed=3)
    pop = np.random.uniform(low=np.array([b[0] for b in bounds]), high=np.array([b[1] for b in bounds]), size=(popsize, len(bounds)))

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    best_fitness = 1

    # Evaluate initial population
    fitness = np.zeros(len(pop))
    avg_acc = 0
    for i in range(len(pop)):
        fitness[i] = obj_func(pop[i],X_train, X_test, y_train, y_test)
        if fitness[i] < best_fitness:
            best_fitness = fitness[i]

    fitness_arr.append(1 - best_fitness)
    
    # Find best solution
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx]
    best_fitness = fitness[best_idx]
    
    # Start evolution
    for i in range(maxiter):
        avg_fitness = 0
        for j in range(popsize):
            # Choose three random individuals, not including the current one
            candidates = np.arange(popsize)
            candidates = np.delete(candidates, j)
            np.random.shuffle(candidates)
            a, b, c = candidates[:3]

            # Mutation
            mutant = pop[a] + mutation * (pop[b] - pop[c])

            # Clip mutant values to bounds
            mutant = np.clip(mutant, 0, 1)

            # Crossover
            crossover = np.random.rand(len(bounds)) < recombination
            if not np.any(crossover):
                crossover[np.random.randint(0, len(bounds))] = True

            # Create trial individual
            trial = np.where(crossover, mutant, pop[j])

            # Clip trial values to bounds
            trial_denorm = min_b + trial * diff
            trial_denorm = np.clip(trial_denorm, min_b, max_b)
            trial = (trial_denorm - min_b) / diff

            # Evaluate trial individual
            f = obj_func(trial_denorm, X_train, X_test, y_train, y_test)
            avg_fitness += 1 - f
            # Update population if trial is better than current individual
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < best_fitness:
                    best_fitness = f
                    best_solution = trial_denorm
        fitness_arr.append(1 - best_fitness)
    
    print("best solution: ", best_solution)
    print("best_fitness: ", 1 - best_fitness)
    return best_solution, best_fitness, fitness_arr

# Define the function to optimize (mean cross-validation score)
def svm_eval(params, X_train, X_test, y_train, y_test):

    # Create a SVM classifier object
    clf = SVC(kernel='rbf', C = params[0], gamma = params[1])

    # Train the SVM classifier
    clf.fit(X_train, y_train)

    # Predict the labels of the test set using the trained SVM classifier
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the SVM classifier on the test set
    accuracy = accuracy_score(y_test, y_pred)

    return 1 - accuracy

def main():
    best_solution, best_fitness, fitness_arr = differential_evolution(svm_eval, bounds = [(0.01, 10), (0.01, 10)])

    df = pd.read_csv("american_bankruptcy_dataset.csv")

    df.loc[:, 'status_label'] = df.loc[:, 'status_label'].eq('failed').mul(1)
    df = df.drop(['year', 'company_name'], axis=1)

    bankrupt_df = df.loc[df["status_label"] == 1]
    non_bankrupt_df = df.loc[df["status_label"] == 0]
    bankrupt_df = bankrupt_df.sample(n = 100, random_state=1)
    non_bankrupt_df = non_bankrupt_df.sample(n = 1000, random_state=1)
    sampled_df = pd.concat([bankrupt_df, non_bankrupt_df], axis=0, ignore_index=True)

    y = sampled_df.iloc[:, 0]

    X = sampled_df.iloc[:, 1:]

    sm = SMOTE(sampling_strategy='auto', k_neighbors=10, random_state=1)
    X_res, y_res = sm.fit_resample(X, y)

    X_res = (X_res-X_res.mean())/X_res.std()

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
    
    # Create a SVM classifier object
    svm = SVC(kernel='rbf', C = best_solution[0], gamma = best_solution[1])

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
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Sensitivity: {sensitivity:.5f}")
    print(f"Specificity: {specificity:.5f}")
    print(f"F-measure: {f_measure:.5f}")
    print(f"Confusion matrix:\n{conf_matrix}")
    print("time: ", end - start) 
    print("AUC: ", auc(fpr, tpr))

    a = np.asarray(fitness_arr)
    np.savetxt("de.csv", a, delimiter=",")

if __name__ == "__main__":
    main()