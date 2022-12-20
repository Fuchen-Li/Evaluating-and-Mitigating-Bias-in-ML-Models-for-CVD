import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler, SMOTE

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

from .optimal_threhold_related import get_optimal_threshold_Jvalue, calculate_tpr, calculate_positive_prediction


def get_EOD(y_test_1, y_score_1,threshold_1, y_test_2, y_score_2, threshold_2):
    """
    calculate equal opportunity difference (difference in true positive rate) across two groups
    """    
    tpr_1 = calculate_tpr(y_test_1, y_score_1, threshold=threshold_1)
    print ("True positive rate of class 1 is " , tpr_1)

    tpr_2 = calculate_tpr(y_test_2, y_score_2, threshold=threshold_2)
    print("True positive rate of class 2 is " , tpr_2)

    eod = tpr_1 - tpr_2
    return eod


def get_SP(y_test_1, y_score_1, threshold_1, y_test_2, y_score_2, threshold_2):
    """
    calculate equal opportunity difference (difference in true positive rate) across two groups
    """
    pd_1 = calculate_positive_prediction(y_test_1, y_score_1, threshold=threshold_1)
    print("Positive prediction rate of class 1 is " , pd_1)

    pd_2 = calculate_positive_prediction(y_test_2, y_score_2, threshold=threshold_2)
    print("Positive prediction rate of class 2 is " , pd_2)

    sp = pd_1/pd_2
    return sp


def balance_accuracy (y_val, y_val_score,y_test, y_test_score):
    
    threshold, _ = thres.get_optimal_threshold_Jvalue (y_val, y_val_score)
    print ("Optimal threshold by J value is ",threshold)

    ba_val = thres.calculate_balanced_accuracy(y_val, y_val_score, threshold)
    print ("Balanced accuracy score of val is ", ba_val)

    ba_test = thres.calculate_balanced_accuracy(y_test, y_test_score, threshold)
    print ("Balanced accuracy score of test is ",ba_test)

    return threshold, ba_val, ba_test

# in this method, we drop the protected attribute
def split_by_trait(X, y, attribute, random_state):
    """get test set"""
    df_train_val, df_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[df_test[attribute]==1]
    y_test_2 = y_test[df_test[attribute]==0]
    
    df_test_1 = df_test[df_test[attribute]==1]
    X_test_1 = df_test_1.drop([attribute], axis=1).values
    df_test_2 = df_test[df_test[attribute]==0]
    X_test_2 = df_test_2.drop([attribute], axis=1).values
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[df_val[attribute]==1]
    y_val_2 = y_val[df_val[attribute]==0]
       
    df_val_1 = df_val[df_val[attribute]==1]
    X_val_1 = df_val_1.drop([attribute], axis=1).values
    df_val_2 = df_val[df_val[attribute]==0]
    X_val_2 = df_val_2.drop([attribute], axis=1).values

    """The overall X set should be protected (exclude the attribute)"""
    X_train = df_train.drop([attribute], axis=1).values
    X_val = df_val.drop([attribute], axis=1).values
    X_test = df_test.drop([attribute], axis=1).values
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_val_1, X_val_2, y_val_1, y_val_2, X_test_1, X_test_2, y_test_1, y_test_2


# In this method, we don't remove the protected attrbute
def split_by_trait_no_protected_trait (X, y, attribute, random_state):
    """get test set"""
    df_train_val, df_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size=0.25, random_state=random_state)
    # overall, we get 20% test, 20% validation, and 60% test
    
    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[df_test[attribute]==1]
    y_test_2 = y_test[df_test[attribute]==0]
    
    df_test_1 = df_test[df_test[attribute]==1]    
    df_test_2 = df_test[df_test[attribute]==0]
        
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[df_val[attribute]==1]
    y_val_2 = y_val[df_val[attribute]==0]
       
    df_val_1 = df_val[df_val[attribute]==1]    
    df_val_2 = df_val[df_val[attribute]==0]
    
    return df_train, y_train, df_val, y_val, df_test, y_test, df_val_1, df_val_2, y_val_1, y_val_2, df_test_1, df_test_2, y_test_1, y_test_2


def split_by_trait_balance_size_removing_protected (X, y, attribute, random_state):
    """get test set"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    # add back the class variable for resampling
    X_train ['Class'] = y_train
    feature_list = list(X_train.columns)

    # balance size of attribute
    X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y = X_train[attribute]
    X = X_train.drop([attribute], axis=1)
    X_over, y_over = RandomOverSampler().fit_resample(X,y)
    y_1 = y_over[y_over == 1]
    y_0 = y_over[y_over == 0]
    print(y_1.shape)
    print(y_0.shape)    

    #add back the feature
    X_over [attribute] = y_over
    # X_over = pd.concat((X_over,y_over),axis=0, names = attribute) # the same as the line above
    
    #shuffle the data
    X_over = X_over.sample(frac = 1)

    # reorder the feature list!
    X_over = X_over[feature_list]
    #print(X_over.head())
    
    resampled_y_train = X_over.Class.values
    resampled_X_train = X_over.drop(['Class'], axis=1)
    print(resampled_X_train.shape)
    
    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[X_test[attribute]==1]
    y_test_2 = y_test[X_test[attribute]==0]
    
    # also remove the protected attribute
    X_test_1 = X_test[X_test[attribute]==1]
    X_test_1 = X_test_1.drop([attribute], axis=1).values
    X_test_2 = X_test[X_test[attribute]==0]
    X_test_2 = X_test_2.drop([attribute], axis=1).values
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[X_val[attribute]==1]
    y_val_2 = y_val[X_val[attribute]==0]
       
    X_val_1 = X_val[X_val[attribute]==1]
    X_val_1 = X_val_1.drop([attribute], axis=1).values
    X_val_2 = X_val[X_val[attribute]==0]
    X_val_2 = X_val_2.drop([attribute], axis=1).values
    
    print(resampled_X_train.head())
    print(X_val.head())
    print(X_test.head())

    """The overall X set should be protected (exclude the attribute)"""
    resampled_X_train = resampled_X_train.drop([attribute], axis=1).values
    X_val = X_val.drop([attribute], axis=1).values
    X_test = X_test.drop([attribute], axis=1).values
    
    return resampled_X_train, resampled_y_train, X_val, y_val, X_test, y_test, X_val_1, X_val_2, y_val_1, y_val_2, X_test_1, X_test_2, y_test_1, y_test_2


def split_by_trait_balance_size_keep_protected(X, y, attribute, random_state):
    # mean race/gender is included in the model
    
    """get test set"""
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    # add back the class variable for resampling
    X_train ['Class'] = y_train
    feature_list = list(X_train.columns)

    # balance size of attribute
    X_train = X_train.replace((np.inf, -np.inf, np.nan), 0).reset_index(drop=True)
    y = X_train[attribute]
    X = X_train.drop([attribute], axis=1)
    X_over, y_over = RandomOverSampler().fit_resample(X,y)
    y_1 = y_over[y_over == 1]
    y_0 = y_over[y_over == 0]
    print("Y=1: " + str(y_1.shape))
    print("Y=0: " + str(y_0.shape))    

    #add back the feature
    X_over [attribute] = y_over
    # X_over = pd.concat((X_over,y_over),axis=0, names = attribute) # the same as the line above
    
    #shuffle the data
    X_over = X_over.sample(frac = 1)

    # reorder the feature list!
    X_over = X_over[feature_list]
    print(X_over.head())
    
    resampled_y_train = X_over.Class.values
    resampled_X_train = X_over.drop(['Class'], axis=1).values
    print("resampled X train shape is" + resampled_X_train.shape)
    
    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[X_test[attribute]==1]
    y_test_2 = y_test[X_test[attribute]==0]
    
    # don't remove the protected attribute
    X_test_1 = X_test[X_test[attribute]==1].values
    X_test_2 = X_test[X_test[attribute]==0].values
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[X_val[attribute]==1]
    y_val_2 = y_val[X_val[attribute]==0]
       
    X_val_1 = X_val[X_val[attribute]==1].values
    X_val_2 = X_val[X_val[attribute]==0].values
    
    return resampled_X_train, resampled_y_train, X_val, y_val, X_test, y_test, X_val_1, X_val_2, y_val_1, y_val_2, X_test_1, X_test_2, y_test_1, y_test_2


def split_by_trait_balance_proportion(X, y, attribute, random_state):
    """get test set"""
    df_train_val, df_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    df_train ['Class'] = y_train
    df_train_0 = df_train[df_train[attribute] == 0]
    """0 for male in Gender, or black in Race_W"""
    df_train_1 = df_train[df_train[attribute] == 1]
    """1 for female in Gender, or white in Race_W"""
    print (df_train_0.shape)
    print (df_train_1.shape)

    df_train_0_affect = df_train_0[df_train_0['Class'] == 1]
    df_train_0_unaffect = df_train_0[df_train_0['Class'] == 0]
    df_train_1_affect = df_train_1[df_train_1['Class'] == 1]
    df_train_1_unaffect = df_train_1[df_train_1['Class'] == 0]
    
    class0_affection_ratio = df_train_0_affect.shape[0]/df_train_0_unaffect.shape[0]
    class1_affection_ratio = df_train_1_affect.shape[0]/df_train_1_unaffect.shape[0]
    print(class0_affection_ratio, class1_affection_ratio)
    higher_affection = max(class0_affection_ratio, class1_affection_ratio)
    lower_affection = min(class0_affection_ratio, class1_affection_ratio)

    frames = []
    if (higher_affection == class0_affection_ratio):
        y = df_train_1.Class
        X = df_train_1.drop(['Class'], axis=1)
        X_over, y_over = SMOTE(sampling_strategy = higher_affection).fit_resample(X,y)
        y_affected = y_over[y_over == 1]
        y_unaffected = y_over[y_over == 0]
        print (y_affected.shape[0]/y_unaffected.shape[0])
        X_over ['Class'] = y_over
        frames = [df_train_0, X_over]
    else:
        y = df_train_0.Class
        X = df_train_0.drop(['Class'], axis=1)
        X_over, y_over = SMOTE(sampling_strategy=higher_affection).fit_resample(X, y)
        y_affected = y_over[y_over == 1]
        y_unaffected = y_over[y_over == 0]
        print (y_affected.shape[0] / y_unaffected.shape[0])
        X_over['Class'] = y_over
        frames = [df_train_1, X_over]
        
    result = pd.concat(frames)
    y_train = result.Class.values
    df_train = result.drop(['Class'], axis=1)
    print (df_train.shape)

    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[df_test[attribute]==1]
    y_test_2 = y_test[df_test[attribute]==0]
    
    df_test_1 = df_test[df_test[attribute]==1]
    X_test_1 = df_test_1.drop([attribute], axis=1).values
    df_test_2 = df_test[df_test[attribute]==0]
    X_test_2 = df_test_2.drop([attribute], axis=1).values
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[df_val[attribute]==1]
    y_val_2 = y_val[df_val[attribute]==0]
       
    df_val_1 = df_val[df_val[attribute]==1]
    X_val_1 = df_val_1.drop([attribute], axis=1).values
    df_val_2 = df_val[df_val[attribute]==0]
    X_val_2 = df_val_2.drop([attribute], axis=1).values

    """The overall X set should be protected (exclude the attribute)"""
    X_train = df_train.drop([attribute], axis=1).values
    X_val = df_val.drop([attribute], axis=1).values
    X_test = df_test.drop([attribute], axis=1).values
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_val_1, X_val_2, y_val_1, y_val_2, X_test_1, X_test_2, y_test_1, y_test_2


def split_by_trait_balance_proportion_no_protected_trait(X, y, attribute, random_state):
    """get test set"""
    df_train_val, df_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    """get train and evaluation set"""
    df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size=0.25, random_state=random_state)
    
    df_train ['Class'] = y_train
    df_train_0 = df_train[df_train[attribute] == 0]
    """0 for male in Gender, or black in Race_W"""
    df_train_1 = df_train[df_train[attribute] == 1]
    """1 for female in Gender, or white in Race_W"""
    print (df_train_0.shape)
    print (df_train_1.shape)

    df_train_0_affect = df_train_0[df_train_0['Class'] == 1]
    df_train_0_unaffect = df_train_0[df_train_0['Class'] == 0]
    df_train_1_affect = df_train_1[df_train_1['Class'] == 1]
    df_train_1_unaffect = df_train_1[df_train_1['Class'] == 0]
    
    class0_affection_ratio = df_train_0_affect.shape[0]/df_train_0_unaffect.shape[0]
    class1_affection_ratio = df_train_1_affect.shape[0]/df_train_1_unaffect.shape[0]
    print(class0_affection_ratio, class1_affection_ratio)
    higher_affection = max(class0_affection_ratio, class1_affection_ratio)
    lower_affection = min(class0_affection_ratio, class1_affection_ratio)

    frames = []
    if (higher_affection == class0_affection_ratio):
        y = df_train_1.Class
        X = df_train_1.drop(['Class'], axis=1)
        X_over, y_over = SMOTE(sampling_strategy = higher_affection).fit_resample(X,y)
        y_affected = y_over[y_over == 1]
        y_unaffected = y_over[y_over == 0]
        print (y_affected.shape[0]/y_unaffected.shape[0])
        X_over ['Class'] = y_over
        frames = [df_train_0, X_over]
    else:
        y = df_train_0.Class
        X = df_train_0.drop(['Class'], axis=1)
        X_over, y_over = SMOTE(sampling_strategy=higher_affection).fit_resample(X, y)
        y_affected = y_over[y_over == 1]
        y_unaffected = y_over[y_over == 0]
        print (y_affected.shape[0] / y_unaffected.shape[0])
        X_over['Class'] = y_over
        frames = [df_train_1, X_over]
        
    result = pd.concat(frames)
    y_train = result.Class.values
    df_train = result.drop(['Class'], axis=1)
    print (df_train.shape)

    """separate the test sets by the trait (attribute) we want"""
    y_test_1 = y_test[df_test[attribute]==1]
    y_test_2 = y_test[df_test[attribute]==0]
    
    df_test_1 = df_test[df_test[attribute]==1]
   
    df_test_2 = df_test[df_test[attribute]==0]
    
    
    """separate the validation sets by the trait (attribute) we want"""
    y_val_1 = y_val[df_val[attribute]==1]
    y_val_2 = y_val[df_val[attribute]==0]
       
    df_val_1 = df_val[df_val[attribute]==1]
    
    df_val_2 = df_val[df_val[attribute]==0]
    
    
    return df_train, y_train, df_val, y_val, df_test, y_test, df_val_1, df_val_2, y_val_1, y_val_2, df_test_1, df_test_2, y_test_1, y_test_2




    
    
    