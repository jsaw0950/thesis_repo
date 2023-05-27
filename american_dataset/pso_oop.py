import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from scipy.io import arff
import random
import time
from imblearn.over_sampling import SMOTE

start = time.time()

def get_data():
    random_state = 42

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


#--- COST FUNCTION ------------------------------------------------------------+

# Define the function to optimize (mean cross-validation score)
def objective_function(params, X_train, X_test, y_train, y_test):

    # Create a SVM classifier object
    clf = SVC(kernel='rbf', C = params[0], gamma=params[1])

    # Train the SVM classifier
    clf.fit(X_train, y_train)

    # Predict the labels of the test set using the trained SVM classifier
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the SVM classifier on the test set
    accuracy = accuracy_score(y_test, y_pred)

    fpr, tpr, threshold = roc_curve(y_test, y_pred)

    return 1- accuracy, clf

#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,init_pos):
        self.position_i=init_pos          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual
        self.best_model = SVC(kernel='rbf')
        np.random.seed(seed=42)

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))

    # evaluate current fitness
    def evaluate(self,costFunc, X_train, X_test, y_train, y_test):
        self.err_i, self.best_model = costFunc(self.position_i, X_train, X_test, y_train, y_test)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i
        

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        np.random.seed(seed=42)
        random.seed(42)
        w=0.5      # constant inertia weight (how much to weigh the previous velocity)
        c1=0.5        # cognative constant
        c2=0.5      # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
class PSO():
    def __init__(self,costFunc,init_pos,bounds,num_particles,maxiter):
        global num_dimensions
        self.X_train, self.X_test, self.y_train, self.y_test = get_data()
        self.best_model = SVC(kernel='rbf')
        self.accuracies = []

        num_dimensions=2
        self.err_best_g=-1                   # best error for group
        self.pos_best_g=[]                   # best position for group

        # # establish the swarm
        # swarm=[]
        # for i in range(0,num_particles):
        #     swarm.append(Particle(x0))

        swarm = []
        for i in init_pos:
            swarm.append(Particle(i))
        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            avg_acc = 0
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc, self.X_train, self.X_test, self.y_train, self.y_test)
                avg_acc += 1 - swarm[j].err_i
                # determine if current particle is the best (globally)
                if swarm[j].err_i < self.err_best_g or self.err_best_g == -1:
                    self.pos_best_g=list(swarm[j].position_i)
                    self.err_best_g=float(swarm[j].err_i)
                    self.best_model = swarm[j].best_model

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(self.pos_best_g)
                swarm[j].update_position(bounds)
            self.accuracies.append(avg_acc/num_particles)
            i+=1

def main():
    bounds=[(0.01,10),(0.01,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    np.random.seed(seed=5)
    init_pos = np.random.uniform(low=np.array([b[0] for b in bounds]), high=np.array([b[1] for b in bounds]), size=(10, len(bounds)))    
    PSO_obj = PSO(objective_function,init_pos, bounds,num_particles=10,maxiter=100)
    best_model = PSO_obj.best_model

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
    svm = SVC(kernel='rbf', C = PSO_obj.pos_best_g[-2], gamma=PSO_obj.pos_best_g[-1])

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

    a = np.asarray(PSO_obj.accuracies)
    np.savetxt("pso.csv", a, delimiter=",")

if __name__ == "__main__":
    main()
