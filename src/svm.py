import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import pprint

################# Given #################
def generate_data(n_samples, tst_frac=0.2, val_frac=0.2):
    X, y = make_moons(n_samples=n_samples, noise=0.25, random_state=42)
    m = 30
    np.random.seed(30)
    ind = np.random.permutation(n_samples)[:m]
    X[ind, :] += np.random.multivariate_normal([0, 0], np.eye(2), (m,))
    y[ind] = 1 - y[ind]
    cmap = ListedColormap(['#b30065', '#178000'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=tst_frac,
                                                  random_state=42)
    X_trn, X_val, y_trn, y_val = train_test_split(X_trn, y_trn, test_size=val_frac,
                                                  random_state=42)
    return (X_trn, y_trn), (X_val, y_val), (X_tst, y_tst)

def visualize(models, param, X, y):
    if len(models) % 3 == 0:
        nrows = len(models) // 3
    else:
        nrows = len(models) // 3 + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
    cmap = ListedColormap(['#b30065', '#178000'])
    xMin, xMax = X[:, 0].min() - 1, X[:, 0].max() + 1
    yMin, yMax = X[:, 1].min() - 1, X[:, 1].max() + 1
    xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01),
                               np.arange(yMin, yMax, 0.01))
    for i, (p, clf) in enumerate(models.items()):
        r, c = np.divmod(i, 3)
        ax = axes[r, c]
        zMesh = clf.decision_function(np.c_[xMesh.ravel(), yMesh.ravel()])
        zMesh = zMesh.reshape(xMesh.shape)
        ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)
        if (param == 'C' and p > 0.0) or (param == 'gamma'):
            ax.contour(xMesh, yMesh, zMesh, colors='k', levels=[-1, 0, 1],
                       alpha=0.5, linestyles=['--', '-', '--'])
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k')
        ax.set_title('{0} = {1}'.format(param, p))

# Generate the data
n_samples = 300
(X_trn, y_trn), (X_val, y_val), (X_tst, y_tst) = generate_data(n_samples)


################# Effect of Regularization Parameter, $C$ #################
print("Begin of Analysis: Effect of Regularization Parameter, $C$")
C_range = np.arange(-3.0, 6.0, 1.0)
C_values = np.power(10.0, C_range)
models = dict()
trnErr = dict()
valErr = dict()
tstErr = dict()
for C in C_values:
    models[C] = SVC(C=C, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr',
                    degree=3, gamma='scale', kernel='rbf',
                    max_iter=-1, probability=False, random_state=None,
                    shrinking=True, tol=0.001, verbose=False)
    fit = models[C].fit(X_trn, y_trn)
    trnErr[C] = 1 - fit.score(X_trn, y_trn)
    valErr[C] = 1 - fit.score(X_val, y_val)
    tstErr[C] = 1 - fit.score(X_tst, y_tst)
visualize(models, 'C', X_trn, y_trn)
bestC = min(valErr, key=valErr.get)
print('*** C-best:', bestC)
yParam = models[bestC].predict(X_tst)
accuracyBestC = accuracy_score(y_tst, yParam)
print("*** Final Test Set Accuracy for C-Best: ", accuracyBestC)
plt.figure()
plt.grid()
plt.xscale('log')
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3, markersize=12)
plt.plot(list(tstErr.keys()), list(tstErr.values()), marker='v', linewidth=3, markersize=12)
plt.xlabel('C', fontsize=12)
plt.ylabel('Validation/Training Error', fontsize=12)
plt.xticks(list(valErr.keys()), fontsize=10)
plt.legend(['Validation Error', 'Training Error'], fontsize=12)
plt.show()
print("End of Analysis: Effect of Regularization Parameter, $C$")
print("..")
print("..")
print("..")
print("..")


################# Effect of RBF Kernel Parameter, $\gamma$ #################
print("Begin of Analysis: Effect of RBF Kernel Parameter, $\gamma$")
gamma_range = np.arange(-2.0, 4.0, 1.0)
gamma_values = np.power(10.0, gamma_range)
models = dict()
trnErr = dict()
valErr = dict()
tstErr = dict()
for G in gamma_values:
    models[G] = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr',
                    degree=3, gamma=G, kernel='rbf',
                    max_iter=-1, probability=False,
                    random_state=None, shrinking=True,
                    tol=0.001, verbose=False)
    fit = models[G].fit(X_trn, y_trn)
    trnErr[G] = 1 - fit.score(X_trn, y_trn)
    valErr[G] = 1 - fit.score(X_val, y_val)
    tstErr[G] = 1 - fit.score(X_tst, y_tst)
visualize(models, 'Gamma', X_trn, y_trn)
bestG = min(valErr, key=valErr.get)
print("*** G-Best:", bestG)
yParam = models[bestG].predict(X_tst)
accuracyBestG = accuracy_score(y_tst, yParam)
print("*** Final Test Set Accuracy for G-Best: ", accuracyBestG)
plt.figure()
plt.grid()
plt.xscale('log')
plt.plot(list(valErr.keys()), list(valErr.values()), marker='o', linewidth=3, markersize=12)
plt.plot(list(trnErr.keys()), list(trnErr.values()), marker='s', linewidth=3, markersize=12)
plt.plot(list(tstErr.keys()), list(tstErr.values()), marker='v', linewidth=3, markersize=12)
plt.xlabel('Gamma', fontsize=12)
plt.ylabel('Validation/Training error', fontsize=10)
plt.xticks(list(valErr.keys()), fontsize=12)
plt.legend(['Validation Error', 'Training Error'], fontsize=12)
plt.show()
print("End of Analysis: Effect of RBF Kernel Parameter, $\gamma$")
print("..")
print("..")
print("..")
print("..")

accuracySVM=0.0
accuracyKNN=0.0
################# Breast Cancer Diagnosis using Support Vector Machines #################
print("Begin of Analysis: Breast Cancer Diagnosis using SVM ")
svm_trn = np.loadtxt('wdbc_trn.csv', delimiter=',')
svm_tst = np.loadtxt('wdbc_tst.csv', delimiter=',')
svm_val = np.loadtxt('wdbc_val.csv', delimiter=',')
X_trn = np.array(svm_trn[:, 1:])
y_trn = np.array(svm_trn[:, 0])
X_tst = np.array(svm_tst[:, 1:])
y_tst = np.array(svm_tst[:, 0])
X_val = np.array(svm_val[:, 1:])
y_val = np.array(svm_val[:, 0])
C_range = np.arange(-2.0, 5.0, 1.0)
C_vals = np.power(10.0, C_range)
gamma_range = np.arange(-3.0, 3.0, 1.0)
gamma_vals = np.power(10.0, gamma_range)
models = dict()
trn_err = dict()
val_err = dict()
tstErr = dict()
accuracyBestC = 0
bestC = 0
bestGammaValue = 0
for C in C_vals:
    for G in gamma_vals:
        models[(C, G)] = SVC(C=C, coef0=0.0, decision_function_shape='ovr', degree=3,
                             gamma=G, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True,
                             tol=0.001, verbose=False)

        fit = models[(C, G)].fit(X_trn, y_trn)
        trn_err[(C, G)] = 1 - fit.score(X_trn, y_trn)
        val_err[(C, G)] = 1 - fit.score(X_val, y_val)
bestC_Gamma = min(val_err, key=val_err.get)
print("*** C-Best, Gamma-Best: ", bestC_Gamma)
yParam = models[bestC_Gamma].predict(X_tst)
accuracyBestC_Gamma = accuracy_score(y_tst, yParam)
print("*** Final Test Set Accuracy for C-Best and Gamma-Best: ", accuracyBestC_Gamma)
accuracySVM=accuracyBestC_Gamma
tables = pprint.PrettyPrinter(indent=2)
print("*** Table: TrainingError")
tables.pprint(trn_err)
print("*** Table: ValidationError")
tables.pprint(val_err)
print("End of Analysis: Breast Cancer Diagnosis using SVM")
print("..")
print("..")
print("..")
print("..")


################# Breast Cancer Diagnosis using K-Nearest Neighbours #################
print("Begin of Analysis: Breast Cancer Diagnosis using K-NN")
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
K_values= np.array([1,5,11,15,21])
trnErr = dict()
valErr = dict()
print("*** K v/s TrainingError , ValidationError")
for i in range(len(K_values)):
    model = KNeighborsClassifier(n_neighbors=K_values[i])
    model.fit(X_trn,y_trn)
    trnErr[i]=1-model.score(X_trn,y_trn)
    valErr[i]=1-model.score(X_val,y_val)
    print(K_values[i],'  ',trnErr[i],' ',valErr[i])
print("*** K v/s TrainingError , ValidationError")
x1,y1=zip(*trnErr.items())
x2,y2=zip(*valErr.items())
plt.figure()
plt.plot(list(K_values), y1, marker='o', linewidth=4, markersize=14)
plt.plot(list(K_values), y2, marker='s', linewidth=4, markersize=14)
plt.xscale('log')
plt.xlabel('K', fontsize=20)
plt.ylabel('Validation/Training error', fontsize=20)
plt.xticks(list(K_values), fontsize=16)
plt.legend(['Validation Error', 'Training Error'], fontsize=20)
plt.axis()
bestVal = min(valErr, key=valErr.get)
print("*** K-Best: ",bestVal)
RegAccuracy=KNeighborsClassifier(n_neighbors=1)
RegAccuracy.fit(X_trn,y_trn)
accuracyBestK=RegAccuracy.score(X_tst,y_tst)
print("*** Final Test Set Accuracy for K-Best: ",accuracyBestK)
accuracyKNN=accuracyBestK
plt.show()
print("End of Analysis: Breast Cancer Diagnosis using K-NN")
print("..")
print("..")
print("..")
print("..")
print("Accuracy SVM: ",accuracySVM)
print("Accuracy KNN: ",accuracyKNN)
if accuracySVM>accuracyKNN:
    print("SVM is better suited for this classification task")
else:
    print("KNN is better suited for this classification task")
