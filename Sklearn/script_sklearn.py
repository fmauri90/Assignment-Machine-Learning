
from sklearn import metrics
from sklearn import datasets
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

#define the SVM function
def rbf_svm(X_train,y_train, X_test, C):
	clf = SVC(C=C, kernel='rbf', class_weight='auto')
	clf.fit(X_train, y_train)
	return clf.predict(X_test)

#define the Random Forest funcion
def estimator_RFC(X_train, y_train, X_test, n_estimators):
	clf = RandomForestClassifier(n_estimators = n_estimators, criterion='gini', random_state=None)
	clf.fit(X_train, y_train)
	return clf.predict(X_test)


#Create a classification dataset (n_samples >= 1000, n_features >= 10) 

dataset_x, dataset_y = datasets.make_classification(n_samples=5000, n_features=50)

n = len(dataset_x)

kf = cross_validation.KFold(n, n_folds=10, shuffle=True, random_state=None)
accuracy_GM = []
f1_GM = []
auc_GM = []
accuracy_SVM = []
f1_SVM = []
auc_SVM = []
accuracy_RFC = []
f1_RFC = []
auc_RFC = []
matrix_bestC = []
matrix_bestEstimator = []

for train_index, test_index in kf:

	X_train, X_test = dataset_x[train_index], dataset_x[test_index]
  	y_train, y_test = dataset_y[train_index], dataset_y[test_index]
	
	#Gaussian NB
	
	clf_gaussian = GaussianNB()
	clf_gaussian.fit(X_train, y_train)
	pred_gaussian = clf_gaussian.predict(X_test)	
	accuracy_GM.append(metrics.accuracy_score(y_test,pred_gaussian)
	f1_GM.append(metrics.f1_score(y_test,pred_gaussian))
	auc_GM.append(metrics.roc_auc_score(y_test,pred_gaussian))
	

	#SVM	
	lengh_X_train = len(X_train)	
	bestC = None
	Cvalues = [1e-02, 1e-01, 1e00, 1e01, 1e02]	
	innerscore_SVM = []
	for C in Cvalues:
		#inner 5-fold cross validationon the original training set for parameter section(C)
		ikf = cross_validation.KFold(lengh_X_train, n_folds=5, shuffle=True, random_state=5000)
		innerf1 = []
		for itrain_index, itest_index in ikf:
			iX_train, iX_test = X_train[itrain_index], X_train[itest_index]
			iy_train, iy_test = y_train[itrain_index], y_train[itest_index]
			ipred = rbf_svm(iX_train, iy_train, iX_test, C)
			#save the F1-score of the inner cross validation
			innerf1.append(metrics.f1_score(iy_test, ipred))
		#compute the avarage
		innerscore_SVM.append(sum(innerf1)/len(innerf1))
	#pick the C that gives best F1-score
	bestC = Cvalues[np.argmax(innerscore_SVM)]
	#predict the labels for the test set using the best C parameter
	pred_SVM = rbf_svm(X_train, y_train, X_test, bestC)
	accuracy_SVM.append(metrics.accuracy_score(y_test,pred_SVM))
	f1_SVM.append(metrics.f1_score(y_test,pred_SVM))
	auc_SVM.append(metrics.roc_auc_score(y_test,pred_SVM))
	matrix_bestC.append(bestC)

	
	
	#Random Forest Classifier	
	lengh_X_train = len(X_train)	
	bestEstimator = None
	Estimatorvalues = [10, 100, 1000]	
	innerscore_RFC = []
	for n_estimators in Estimatorvalues:
		#inner 5-fold cross validationon the original training set for parameter section(n_estimators)
		ikf = cross_validation.KFold(lengh_X_train, n_folds=5, shuffle=True, random_state=5000)
		innerf1 = []
		for itrain_index, itest_index in ikf:
			iX_train, iX_test = X_train[itrain_index], X_train[itest_index]
			iy_train, iy_test = y_train[itrain_index], y_train[itest_index]
			ipred = estimator_RFC(iX_train, iy_train, iX_test, n_estimators)
			#save the F1-score of the inner cross validation
			innerf1.append(metrics.f1_score(iy_test, ipred))
		#compute the avarage
		innerscore_RFC.append(sum(innerf1)/len(innerf1))
	#pick the estimator that gives best F1-score
	bestEstimator = Estimatorvalues[np.argmax(innerscore_RFC)]
	#predict the labels for the test set using the best C parameter
	pred_RFC = estimator_RFC(X_train, y_train, X_test, bestEstimator)
	accuracy_RFC.append(metrics.accuracy_score(y_test,pred_RFC))
	f1_RFC.append(metrics.f1_score(y_test,pred_RFC))
	auc_RFC.append(metrics.roc_auc_score(y_test,pred_RFC))
	matrix_bestEstimator.append(bestEstimator)


print ("bestC", matrix_bestC)
print ("bestEstimator", matrix_bestEstimator)



print ("accuracy_gaussian", accuracy_GM)
print ("f1_gaussian", f1_GM)
print ("auc_gaussian", auc_GM)
print ("accuracy_SVM", accuracy_SVM)
print ("f1_SVM", f1_SVM)
print ("auc_SVM", auc_SVM)
print ("accuracy_RFC", accuracy_RFC)
print ("f1_RFC", f1_RFC)
print ("auc_RFC", auc_RFC)



#accuracy

#Calculate the mean of the accuracy of GaussianNB
mean_accuracy_GM = np.mean(accuracy_GM)
#Calculate the max of the accuracy of GaussianNB
max_accuracy_GM = np.max(accuracy_GM)
#Calculate the min of the accuracy of GaussianNB
min_accuracy_GM = np.min(accuracy_GM)
#Calculate the standard deviation of accuracy of GaussianNB
std_accuracy_GM = np.std(accuracy_GM)


#print the mean of the accuracy of GaussianNB
print ("mean of accuracy of GaussianNB", mean_accuracy_GM)
#print the max of the accuracy of GaussianNB
print ("max of accuracy of GaussianNB", max_accuracy_GM)
#print the min of the accuracy of GaussianNB
print ("min of accuracy of GaussianNB", min_accuracy_GM)
#print the standard deviation of accuracy of GaussianNB
print ("standard deviation of accuracy of GaussianNB", std_accuracy_GM)


#Calculate the mean of the accuracy of SVM
mean_accuracy_SVM = np.mean(accuracy_SVM)
#Calculate the max of the accuracy of SVM
max_accuracy_SVM = np.max(accuracy_SVM)
#Calculate the min of the accuracy of SVM
min_accuracy_SVM = np.min(accuracy_SVM)
#Calculate the standard deviation of accuracy of SVM
std_accuracy_SVM = np.std(accuracy_SVM)

#print the mean of the accuracy of SVM
print ("mean of accuracy of SVM", mean_accuracy_SVM)
#print the max of the accuracy of SVM
print ("max of accuracy of SVM", max_accuracy_SVM)
#print the min of the accuracy of SVM
print ("min of accuracy of SVM", min_accuracy_SVM)
#print the standard deviation of accuracy of SVM
print ("standard deviation of accuracy of SVM", std_accuracy_SVM)

#Calculate the mean of the accuracy of Random Forest Classifier
mean_accuracy_RFC = np.mean(accuracy_RFC)
#Calculate the max of the accuracy of Random Forest Classifier
max_accuracy_RFC = np.max(accuracy_RFC)
#Calculate the min of the accuracy of Random Forest Classifier
min_accuracy_RFC = np.min(accuracy_RFC)
#Calculate the standard deviation of the accuracy of Random Forest Classifier
std_accuracy_RFC = np.std(accuracy_RFC)

#print the mean of the accuracy of Random Forest Classifier
print ("mean of accuracy of Random Forest Classifier", mean_accuracy_RFC)
#print the max of the accuracy of Random Forest Classifier
print ("max of accuracy of Random Forest Classifier", max_accuracy_RFC)
#print the min of the accuracy of Random Forest Classifier
print ("min of accuracy of Random Forest Classifier", min_accuracy_RFC)
#print the standard deviation of the accuracy of Random Forest Classifier
print ("standard deviation accuracy of Random Forest Classifier", std_accuracy_RFC)




#f1


#Calculate the mean of the f1 of GaussianNB
mean_f1_GM = np.mean(f1_GM)
#Calculate the max of the f1 of GaussianNB
max_f1_GM = np.max(f1_GM)
#Calculate the min of the f1 of GaussianNB
min_f1_GM = np.min(f1_GM)
#Calculate the standard deviation of f1 of GaussianNB
std_f1_GM = np.std(f1_GM)


#print the mean of the f1 of GaussianNB
print ("mean of f1 of GaussianNB", mean_f1_GM)
#print the max of the f1 of GaussianNB
print ("max of f1 of GaussianNB", max_f1_GM)
#print the min of the f1 of GaussianNB
print ("min of f1 of GaussianNB", min_f1_GM)
#print the standard deviation of the mean of f1 of GaussianNB
print ("standard deviation of f1 of GaussianNB", std_f1_GM)


#Calculate the mean of the f1 of SVM
mean_f1_SVM = np.mean(f1_SVM)
#Calculate the max of the f1 of SVM
max_f1_SVM = np.max(f1_SVM)
#Calculate the min of the f1 of SVM
min_f1_SVM = np.min(f1_SVM)
#Calculate the standard deviation of the f1 of SVM
std_f1_SVM = np.std(f1_SVM)

#print the mean of the f1 of SVM
print ("mean of f1 of SVM", mean_f1_SVM)
#print the max of the f1 of SVM
print ("max of f1 of SVM", max_f1_SVM)
#print the min of the f1 of SVM
print ("min of f1 of SVM", min_f1_SVM)
#print the standard deviation of the mean of the f1 of SVM
print ("standard deviation of f1 of SVM", std_f1_SVM)

#Calculate the mean of the f1 of Random Forest Classifier
mean_f1_RFC = np.mean(f1_RFC)
#Calculate the max of the f1 of Random Forest Classifier
max_f1_RFC = np.max(f1_RFC)
#Calculate the min of the f1 of Random Forest Classifier
min_f1_RFC = np.min(f1_RFC)
#Calculate the standard deviation of the f1 of Random Forest Classifier
std_f1_RFC = np.std(f1_RFC)

#print the mean of the f1 of Random Forest Classifier
print ("mean of f1 of Random Forest Classifier", mean_f1_RFC)
#print the max of the f1 of Random Forest Classifier
print ("max of f1 of Random Forest Classifier", max_f1_RFC)
#print the min of the f1 of Random Forest Classifier
print ("min of f1 of Random Forest Classifier", min_f1_RFC)
#print the standard deviation of the f1 of Random Forest Classifier
print ("standard deviation of f1 of Random Forest Classifier", std_f1_RFC)




#auc

#Calculate the mean of the auc of GaussianNB
mean_auc_GM = np.mean(auc_GM)
#Calculate the max of the auc of GaussianNB
max_auc_GM = np.max(auc_GM)
#Calculate the min of the auc of GaussianNB
min_auc_GM = np.min(auc_GM)
#Calculate the standard deviation of auc of GaussianNB
std_auc_GM = np.std(auc_GM)


#print the mean of the auc of GaussianNB
print ("mean of auc of GaussianNB", mean_auc_GM)
#print the max of the auc of GaussianNB
print ("max of auc of GaussianNB", max_auc_GM)
#print the min of the auc of GaussianNB
print ("min of auc of GaussianNB", min_auc_GM)
#print the standard deviation of auc of GaussianNB
print ("standard deviation of auc of GaussianNB", std_auc_GM)


#Calculate the mean of the auc of SVM
mean_auc_SVM = np.mean(auc_SVM)
#Calculate the max of the auc of SVM
max_auc_SVM = np.max(auc_SVM)
#Calculate the min of the auc of SVM
min_auc_SVM = np.min(auc_SVM)
#Calculate the standard deviation of the auc of SVM
std_auc_SVM = np.std(auc_SVM)

#print the mean of the auc of SVM
print ("mean of auc of SVM", mean_auc_SVM)
#print the max of the auc of SVM
print ("max of auc of SVM", max_auc_SVM)
#print the min of the auc of SVM
print ("min of auc of SVM", min_auc_SVM)
#print the standard deviation of the auc of SVM
print ("standard deviation of auc of SVM", std_auc_SVM)

#Calculate the mean of the auc of Random Forest Classifier
mean_auc_RFC = np.mean(auc_RFC)
#Calculate the max of the auc of Random Forest Classifier
max_auc_RFC = np.max(auc_RFC)
#Calculate the min of the auc of Random Forest Classifier
min_auc_RFC = np.min(auc_RFC)
#Calculate the standard deviation of the auc of Random Forest Classifier
std_auc_RFC = np.std(auc_RFC)

#print the mean of the auc of Random Forest Classifier
print ("mean of auc of Random Forest Classifier", mean_auc_RFC)
#print the max of the auc of Random Forest Classifier
print ("max of auc of Random Forest Classifier", max_auc_RFC)
#print the min of the auc of Random Forest Classifier
print ("min of auc of Random Forest Classifier", min_auc_RFC)
#print the standard deviation of the auc of Random Forest Classifier
print ("standard deviation of Random Forest Classifier", std_auc_RFC)
