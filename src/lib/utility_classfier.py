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


def rank_features(clf, feature_list, top_feature=10, abs=False):
    """
    :param clf:
    :param feature_list:
    :param top_feature:
    :param abs: if rank with the absolute value of the coefficient
    :return:
    """
    selected_features_list = []
    if hasattr(clf, 'coef_'):
        print(clf.coef_.shape)
        if abs is True:
            abs_coef = np.abs(clf.coef_[0])
            indices = np.argsort(abs_coef)[::-1]  # ordered from largest to small
        else:
            indices = np.argsort(clf.coef_[0])[::-1]  # ordered from largest to small

        for f in range(top_feature):
            selected_features_list.append({
                'feature_name': '%s' % feature_list[indices[f]],
                'importance': '%.4f' % clf.coef_[0][indices[f]],
                'index': '%d' % indices[f]
            })
            print("%d. feature %s (%f)" % (f + 1, feature_list[indices[f]], clf.coef_[0][indices[f]]))

    elif hasattr(clf, 'feature_importances_'):
        indices = np.argsort(clf.feature_importances_)[::-1]  # ordered from largest to small
        for f in range(top_feature):
            selected_features_list.append({
                'feature_name': '%s' % feature_list[indices[f]],
                'importance': '%.4f' % clf.feature_importances_[indices[f]],
                'index': '%d' % indices[f]
            })
            print("%d. feature %s (%f)" % (f + 1, feature_list[indices[f]], clf.feature_importances_[indices[f]]))

    return selected_features_list


# logic regression
def logic_regression(X_train, y_train, X_valid, y_valid, c=10, feature_list=None, top_features_num=20, dump_model = False, file_name=""):
    t0 = time()

    clf = LogisticRegression(solver = 'saga', penalty = 'l1', l1_ratio = 0.8)
    clf = clf.fit(X_train, y_train)
    
    if dump_model:
        pickle.dump(clf, open(file_name, 'wb'))
        
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
def random_forest(X_train, y_train, X_valid, y_valid, feature_list=None,top_features_num=20, bootstrap=True, dump_model = False, file_name=""):
    t0 = time()
       
    clf = ensemble.RandomForestClassifier(n_estimators = 50, max_features = 'sqrt', max_depth = 8, criterion = 'entropy')   
    clf = clf.fit(X_train, y_train)
        
    if dump_model:
        pickle.dump(clf, open(file_name, 'wb'))
        
    y_pred = clf.predict(X_valid)
    # np.savetxt("random.csv", y_pred.astype(int), fmt='%i', delimiter=",")
    print("Classification report")
    print(classification_report(y_valid, y_pred))
    print("Confusion_matrix")
    print(confusion_matrix(y_valid, y_pred))
    print("done in %fs" % (time() - t0))
    y_score = clf.predict_proba(X_valid)[:, 1]
    #if feature_list is not None:
        #selected_features = rank_features(clf, feature_list, top_features_num)
    return clf


def gradiant_boosting(X_train, y_train, X_valid, y_valid, feature_list=None,top_features_num=20, dump_model = False, file_name=""):
    t0 = time()
    
    clf = gbc(max_features = 9, max_depth = 8, n_estimators = 500, learning_rate = 0.01, subsample = 0.9)    
    clf = clf.fit(X_train, y_train)
    
    if dump_model:
        pickle.dump(clf, open(file_name, 'wb'))

    y_pred = clf.predict(X_valid)
    # np.savetxt("random.csv", y_pred.astype(int), fmt='%i', delimiter=",")
    print("Classification report")
    print(classification_report(y_valid, y_pred))
    print("Confusion_matrix")
    print(confusion_matrix(y_valid, y_pred))
    print("done in %fs" % (time() - t0))
    y_score = clf.predict_proba(X_valid)[:, 1]
    #if feature_list is not None:
        #rank_features(clf, feature_list, top_features_num)
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


