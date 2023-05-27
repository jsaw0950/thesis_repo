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

    d1= arff.loadarff('../5year.arff')

    df = pd.DataFrame(d1[0])

    df.drop_duplicates(inplace=True)

    df.dropna(thresh=50,inplace=True)

    imr = SimpleImputer(missing_values=np.nan, strategy='mean')

    imr=imr.fit(df)

    imputed_data=imr.transform(df.values)

    imputed_data_df=pd.DataFrame(imputed_data)

    df=pd.DataFrame(imputed_data_df.values,columns=["Attr1","Attr2","Attr3","Attr4","Attr5","Attr6","Attr7","Attr8","Attr9","Attr10","Attr11","Attr12","Attr13","Attr14","Attr15","Attr16","Attr17","Attr18","Attr19","Attr20","Attr21","Attr22","Attr23","Attr24","Attr25","Attr26","Attr27","Attr28","Attr29","Attr30","Attr31","Attr32","Attr33","Attr34","Attr35","Attr36","Attr37","Attr38","Attr39","Attr40","Attr41","Attr42","Attr43","Attr44","Attr45","Attr46","Attr47","Attr48","Attr49","Attr50","Attr51","Attr52","Attr53","Attr54","Attr55","Attr56","Attr57","Attr58","Attr59","Attr60","Attr61","Attr62","Attr63","Attr64","Class"])

    bankrupt_df = df.loc[df["Class"] == 1]
    non_bankrupt_df = df.loc[df["Class"] == 0]
    bankrupt_df = bankrupt_df.sample(n = 100, random_state=42)
    non_bankrupt_df = non_bankrupt_df.sample(n = 1000, random_state=42)
    sampled_df = pd.concat([bankrupt_df, non_bankrupt_df], axis=0, ignore_index=True)

    y = sampled_df.iloc[:,-1]

    X = sampled_df.iloc[:, :-1]

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
    
    # Evaluate initial population
    fitness = np.zeros(len(pop))
    avg_acc = 0
    for i in range(len(pop)):
        fitness[i] = obj_func(pop[i],X_train, X_test, y_train, y_test)
        avg_acc += 1 - fitness[i]

    fitness_arr.append(avg_acc/len(pop))
    
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
        print(1 - best_fitness, time.time() - start)
        fitness_arr.append(1 - best_fitness)
    
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

    d1= arff.loadarff('../5year.arff')

    df = pd.DataFrame(d1[0])

    df.drop_duplicates(inplace=True)

    df.dropna(thresh=50,inplace=True)

    imr = SimpleImputer(missing_values=np.nan, strategy='mean')

    imr=imr.fit(df)

    imputed_data=imr.transform(df.values)

    imputed_data_df=pd.DataFrame(imputed_data)

    df=pd.DataFrame(imputed_data_df.values,columns=["Attr1","Attr2","Attr3","Attr4","Attr5","Attr6","Attr7","Attr8","Attr9","Attr10","Attr11","Attr12","Attr13","Attr14","Attr15","Attr16","Attr17","Attr18","Attr19","Attr20","Attr21","Attr22","Attr23","Attr24","Attr25","Attr26","Attr27","Attr28","Attr29","Attr30","Attr31","Attr32","Attr33","Attr34","Attr35","Attr36","Attr37","Attr38","Attr39","Attr40","Attr41","Attr42","Attr43","Attr44","Attr45","Attr46","Attr47","Attr48","Attr49","Attr50","Attr51","Attr52","Attr53","Attr54","Attr55","Attr56","Attr57","Attr58","Attr59","Attr60","Attr61","Attr62","Attr63","Attr64","Class"])
    
    bankrupt_df = df.loc[df["Class"] == 1]
    non_bankrupt_df = df.loc[df["Class"] == 0]
    bankrupt_df = bankrupt_df.sample(n = 100, random_state=1)
    non_bankrupt_df = non_bankrupt_df.sample(n = 1000, random_state=1)
    sampled_df = pd.concat([bankrupt_df, non_bankrupt_df], axis=0, ignore_index=True)

    y = sampled_df.iloc[:,-1]

    X = sampled_df.iloc[:, :-1]

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
    roc_auc = auc(fpr, tpr)
    end = time.time()

    # import matplotlib.pyplot as plt
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

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