import time
import numpy as np
import pandas as pd
import os
from uuid import uuid4
import graphviz
import xgboost as xgb

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn import tree

import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

traindf = spark.read.csv('/tmp/train.csv',header=True,inferSchema=True)
traindf = traindf.toPandas()

testdf = spark.read.csv('/tmp/test.csv',header=True,inferSchema=True)
testdf = testdf.toPandas()

index_train = traindf['id']
index_test = testdf['id']

target = 'status_group_code'
inputCols=['amount_tsh_disc','date_recorded_duration_disc','date_recorded_month_code','funder_code','gps_height_disc','installer_code','basin_code','region_code','district_code','lga_code','population_disc','public_meeting_true','public_meeting_false','scheme_management_code','construction_year_disc','permit_true','permit_false','extraction_type_class_code','management_code','payment_type_code','water_quality_code','quantity_code','source_code','waterpoint_type_code','geo_code']

X_train = traindf[inputCols]
y_train = traindf[target]
X_test = testdf[inputCols]

xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test)

def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgb_param['num_class'] = 3
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['auc'], early_stopping_rounds=early_stopping_rounds, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
	
#Choose all predictors except target & IDcols
xgb1 = XGBClassifier(
 learning_rate =0.001,
 n_estimators=10000,
 max_depth=12,
 min_child_weight=7,
 gamma=0,
 subsample=0.9,
 colsample_bytree=.5,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, traindf, inputCols)

#Parameters with highest accuracy
param_test6 = {
 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator = XGBClassifier(
 learning_rate =0.2,
 n_estimators=100,
 max_depth=12,
 min_child_weight=7,
 gamma=0,
 subsample=0.9,
 colsample_bytree=.5,
 objective= 'multi:softmax',
 nthread=4,
 scale_pos_weight=1,
 seed=27), 
 param_grid = param_test6, scoring='accuracy',n_jobs=4,iid=False, cv=5)
gsearch6.fit(traindf[inputCols],traindf[target])
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_

#Output the result
result = xgb1.predict(X_test)
predictions = pd.Series(result, index=index_test)
predictions.name = 'status_group'
di = {0: "functional", 1: "non functional", 2: "functional needs repair"}
predictions = predictions.reset_index().replace({'status_group':di})
predictions.to_csv('/tmp/TaarifaResults_V10_XGboost.csv',index=False)