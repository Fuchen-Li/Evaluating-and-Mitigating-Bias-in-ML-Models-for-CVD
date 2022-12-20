import numpy as np
from time import time

from sklearn import metrics, model_selection
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

from sklearn import svm

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.ensemble import AdaBoostClassifier as adb

import pickle


# logic regression
def logic_regression(X_train, y_train, dump_model = False, file_name=""):
    t0 = time()
    clf = LogisticRegression()
    
    solvers = ['saga']
    penalty = ['elasticnet','l1']
    #c_values = [10, 1.0, 0.1]
    l1_ratios = [0.1, 0.5, 0.8]
    grid = dict(solver=solvers,penalty=penalty,l1_ratio=l1_ratios)
    grid_search = GridSearchCV(estimator=clf, param_grid=grid, scoring='roc_auc')
    grid_result = grid_search.fit(X_train, y_train)
    param_dic = grid_result.best_params_
    
    print(param_dic)
    with open("/Users/lifuchen/Desktop/Evaluating-and-Mitigating-Bias-in-ML-Models-for-CVD/Models/parameters.txt", 'w') as f: 
        for key, value in param_dic.items(): 
            f.write('%s:%s\n' % (key, value))
    
    clf = LogisticRegression(solver = param_dic['solver'], penalty = param_dic['penalty'], l1_ratio = param_dic['l1_ratio'])
    clf = clf.fit(X_train, y_train)
    
    if dump_model:
        pickle.dump(clf, open(file_name, 'wb'))
    
    return clf
        


#decision tree
def decision_tree(X_train, y_train, X_valid, y_valid, c=10, feature_list=None, top_features_num=20):
    t0 = time()
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    p_train = clf.predict_proba(X_train)
    p_valid = clf.predict_proba(X_valid)
    y_score = clf.predict_proba(X_valid)[:, 1]

    y_predict_valid = clf.predict(X_valid)
    print(metrics.log_loss(y_train, p_train))
    print(metrics.log_loss(y_valid, p_valid))
    print("Classification report")
    print(classification_report(y_valid, y_predict_valid))
    print("Confusion_matrix")
    print(confusion_matrix(y_valid, y_predict_valid))
    print("done in %fs" % (time() - t0))
    #if feature_list is not None:
        #rank_features(clf, feature_list, top_features_num)
    return y_score


# random forest tree
def random_forest(X_train, y_train, dump_model = False, file_name=""):
    t0 = time()
    
    rfc = ensemble.RandomForestClassifier()
    parameters = { 
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,6,8],
    'criterion' :['gini', 'entropy']
    }
    CV_rfc = RandomizedSearchCV(estimator = rfc, param_distributions = parameters, n_iter = 10, scoring='roc_auc', random_state=0)
    CV_rfc.fit(X_train, y_train)
    param_dic = CV_rfc.best_params_
    print(param_dic)
    with open("/Users/lifuchen/Desktop/Evaluating-and-Mitigating-Bias-in-ML-Models-for-CVD/Models/parameters.txt", 'w') as f: 
        for key, value in param_dic.items(): 
            f.write('%s:%s\n' % (key, value))
    clf = ensemble.RandomForestClassifier(n_estimators = param_dic['n_estimators'], max_features = param_dic['max_features'], max_depth = param_dic['max_depth'], criterion = param_dic['criterion'])
    
    #clf = ensemble.RandomForestClassifier(n_estimators=100, bootstrap=True, criterion='entropy', max_features=10, min_samples_split=10)
    
    clf = clf.fit(X_train, y_train)
        
    if dump_model:
        pickle.dump(clf, open(file_name, 'wb'))
    
    return clf


def gradiant_boosting(X_train, y_train, dump_model = False, file_name=""):
    t0 = time()
    clf = gbc()
    parameters = {"max_features" : [5,9,15],
                  "learning_rate": [0.01, 0.05, 0.1],
                  "subsample"    : [0.1, 0.6, 0.9],
                  "n_estimators" : [50, 100, 500],
                  "max_depth"    : [3,6,9]
                 }
    randm = RandomizedSearchCV(estimator = clf, param_distributions = parameters, n_iter = 10, scoring='roc_auc', random_state=0)
    randm.fit(X_train, y_train)
    param_dic = randm.best_params_
    print(param_dic)
    with open("/Users/lifuchen/Desktop/Evaluating-and-Mitigating-Bias-in-ML-Models-for-CVD/Models/parameters.txt", 'w') as f: 
        for key, value in param_dic.items(): 
            f.write('%s:%s\n' % (key, value))
    clf = gbc(max_features = param_dic['max_features'], max_depth = param_dic['max_depth'], n_estimators = param_dic['n_estimators'], learning_rate = param_dic['learning_rate'], subsample = param_dic['subsample'])
    
    clf = clf.fit(X_train, y_train)
    
    if dump_model:
        pickle.dump(clf, open(file_name, 'wb'))

    return clf


def ada_boosting(X_train, y_train, X_valid, y_valid,feature_list=None):
    t0 = time()
    clf = adb()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    # np.savetxt("random.csv", y_pred.astype(int), fmt='%i', delimiter=",")
    print("Classification report")
    print(classification_report(y_valid, y_pred))
    print("Confusion_matrix")
    print(confusion_matrix(y_valid, y_pred))
    print("done in %fs" % (time() - t0))
    y_score = clf.predict_proba(X_valid)[:, 1]
    if feature_list is not None:
        rank_features(clf, feature_list, 20)
    return y_score

def xgb_boosting(X_train, y_train, X_test, y_test,num_round=2):
    import xgboost as xgb
    from xgboost import XGBClassifier
    xg = XGBClassifier(learning_rate=0.02,
                  objective='binary:logistic',
                  min_child_weight=5,
                  gamma=2,
                  subsample=0.6,
                  colsample_bytree=0.3,
                  max_depth=6,
                  random_state=1234)
    xg.fit(X_train, y_train)
    #param = {'learning_rate': 0.02, 'objective': 'binary:logistic', 'min_child_weight': 5, 'gamma': 2, 'subsample': 0.6, 'colsample_bytree': 0.3, 'max_depth':6, 'random_state':1234}
    #xgtrain = xgb.DMatrix(X_train, y_train)
    #xgtest = xgb.DMatrix(X_test)
    #preds = xgb.predict(X_test)
    #bst = xgb.train(param, xgtrain, num_round)
    # make prediction
    
    y_score = xg.predict_proba(X_test)[:, 1]
    print("Classification report")
    print(classification_report(y_test, preds))
    print("Confusion_matrix")
    print(confusion_matrix(y_test, preds))
    return y_score


def knn(X_train, y_train, X_valid, y_valid, feature_list=None):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train,y_train)
    y_pred = neigh.predict(X_valid)
    print("Classification report")
    print(classification_report(y_valid, y_pred))
    print("Confusion_matrix")
    print(confusion_matrix(y_valid, y_pred))
    y_score = neigh.predict_proba(X_valid)[:, 1]
    if feature_list is not None:
        rank_features(neigh, feature_list, 20)
    return y_score



def my_svm(X_train, y_train, X_test, y_test, kernel):
    t0 = time()
    #     _train = preprocessing.normalize(X_train, norm='l2')
    #     _test = preprocessing.normalize(X_test, norm='l2')
    clf = svm.SVC(kernel=kernel, C=1, probability=True).fit(X_train, y_train)
    y_score_svm = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    # np.savetxt("svm.csv", y_pred.astype(int), fmt='%i', delimiter=",")
    # np.savetxt("svmm.csv", y_score_svm, delimiter=",")
    print(classification_report(y_test, y_pred))
    print("done in %fs" % (time() - t0))
    return y_score_svm



def compute_roc(y_test, y_score, method):
    fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    # Compute micro-average ROC curve and ROC area
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=method + ' (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC ' + method)
    plt.legend(loc="lower right")
    plt.show()


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score


def plot_prc(y_test, y_score, ratio):
    average_precision = average_precision_score(y_test, y_score)
    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    y = ratio
    plt.axhline(y=y, color='navy', linestyle='--')
    # plt.plot([0, y], color='navy', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.4f}'.format(
        average_precision))



