import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('creditcard_csv.csv')
df_cols = df.columns
# fraud_df = df.loc[df["Class"] == 1]
df = df.sample(n=50000)
# df = pd.concat([sampled_df, fraud_df])
df.drop_duplicates(inplace=True)
imr = SimpleImputer(missing_values=np.nan, strategy='mean')
imr=imr.fit(df)
imputed_data=imr.transform(df.values)
imputed_data_df=pd.DataFrame(imputed_data)
df=pd.DataFrame(imputed_data_df.values, columns=df_cols)

scaler = MinMaxScaler()
sampled_df = scaler.fit_transform(df)
sampled_df = pd.DataFrame(sampled_df, columns=df_cols)

X = sampled_df.iloc[:, :-1]
y = sampled_df.iloc[:, -1]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state = 11)

# Define the search space for the SVM hyperparameters
bounds = [(0.01, 10), (0.01, 10)]

# Define the function to optimize (mean cross-validation score)
def objective_function(params, X_train, X_test, y_train, y_test):

    # Create a SVM classifier object
    clf = SVC(kernel='rbf', C = params[0], gamma = params[1])

    # Train the SVM classifier
    clf.fit(X_train, y_train)

    # Predict the labels of the test set using the trained SVM classifier
    y_pred = clf.predict(X_test)

    # Calculate the accuracy of the SVM classifier on the test set
    accuracy = accuracy_score(y_test, y_pred)

    return 1 - accuracy, clf

# Set up the PSO optimizer
num_particles = 5
max_iterations = 10
w = 1.0
c1 = 0.8
c2 = 0.8

np.random.seed(seed=42)
# Initialize particles randomly
particles = np.random.uniform(low=np.array([b[0] for b in bounds]), high=np.array([b[1] for b in bounds]), size=(num_particles, len(bounds)))

# Initialize best particle and score
best_particle = None
best_score = np.inf
best_model = SVC(kernel='rbf')

# Run the optimization
fitness_arr = []
for i in range(max_iterations):
    # Evaluate each particle
    for j in range(num_particles):
        score, model = objective_function(particles[j], Xtrain, Xtest, ytrain, ytest)
        
        fitness_arr.append(1 - score)
        if score < best_score:
            best_particle = particles[j]
            best_score = score
            best_model = model

    # Update particle velocities and positions
    for j in range(num_particles):
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        v_j = w * particles[j] + c1 * r1 * (best_particle - particles[j]) + c2 * r2 * (particles[j] - best_particle)
        particles[j] = np.clip(v_j, [b[0] for b in bounds], [b[1] for b in bounds])
        # maskl = [particles[j] < bounds[i][0] for i in range(len(particles[j]))]
        # masku = [particles[j] > bounds[i][1] for i in range(len(particles[j]))]


# Predict the labels of the test set using the trained SVM classifier
y_pred = best_model.predict(Xtest)

# Compute the performance metrics

conf_matrix = confusion_matrix(ytest, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
specificity = tn / (tn + fp)
accuracy = accuracy_score(ytest, y_pred)
precision = tp / (tp + fp)
sensitivity = tp / (tp + fn)
f_measure = (2*precision*sensitivity) / (precision + sensitivity)
# Print the results
print(f"Accuracy: {accuracy:.5f}")
print(f"Precision: {precision:.5f}")
print(f"Sensitivity: {sensitivity:.5f}")
print(f"Specificity: {specificity:.5f}")
print(f"F-measure: {f_measure:.5f}")
print(f"Confusion matrix:\n{conf_matrix}")

a = np.asarray(fitness_arr)
np.savetxt("pso.csv", a, delimiter=",")