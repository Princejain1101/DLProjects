import os
import pandas as pd
import numpy as np
import matplotlib
import pickle
import xgboost as xgb
from IPython.display import display, HTML
pd.set_option('display.max_columns', None)
print('loaded modules')

Dict = {}

def createDict(train, cols):
    for col in cols:
        navalue = train[col].value_counts().head(1).index[0]
        train[col] = train[col].fillna(navalue)
        dcol = dict(enumerate(pd.Categorical(train[col]).categories))
        dcolu = dict(zip(dcol.values(), dcol.keys()))
        Dict[col] = dcolu
        #print(col,dcolu)

def loadtrain(trainfile):
    print('loading train')
    train = pd.read_csv(trainfile)
    return train

def loadtest(testfile):
    print('loading test')
    test = pd.read_csv(testfile)
    return test

def loadsamplesub(subfile):
    print('loading sample sub')
    sample_sub = pd.read_csv(subfile)
    return sample_sub

def getaddedClass(a,b):
    for i in range(len(a)):
        if(b[i]>a[i]):
            return i
    return -1

def getaddedClass2(a,b):
    adf = pd.DataFrame(a)
    bdf = pd.DataFrame(b)
    reslist = adf.index[bdf[0]>adf[0]].tolist()
    if len(reslist)>0:
        return reslist[0]
    else:
        return -1
     
def getTrainuser(trainuser):
    remcols = train.filter(regex="ind_.*_ult1").columns.tolist()
    trainex = trainuser[remcols]
    userindex = trainuser.index.tolist()
    for i in range(trainex.shape[0]-1):
        p = getaddedClass(trainex.iloc[i].tolist(), trainex.iloc[i+1].tolist())
        trainuser.loc[userindex[i],'sol'] = p

def getTrain3(trainsample):
    remcols = trainsample.filter(regex="ind_.*_ult1").columns.tolist()
    rows = trainsample.shape[0]
    indices = trainsample.index.tolist()
    for i in range(rows):
        ncod = trainsample.iloc[i].ncodpers
        temp = trainsample.iloc[:i][trainsample.iloc[:i].ncodpers==ncod].tail(1)
        if temp.shape[0] == 1:
            p = getaddedClass2(trainsample.iloc[i][remcols].tolist(), temp[remcols].tolist())
            trainsample.loc[indices[i],'sol'] = p
        else:
            trainsample.loc[indices[i],'sol'] = -1
        if i%100==0:
            print i

def getTrain2(trainsample):
    remcols = trainsample.filter(regex="ind_.*_ult1").columns.tolist()
    rows = trainsample.shape[0]
    indices = trainsample.index.tolist()
    for i in range(rows-1):
        if trainsample.iloc[i].ncodpers == trainsample.iloc[i+1].ncodpers:
            p = getaddedClass2(trainsample.iloc[i][remcols].tolist(), trainsample.iloc[i+1][remcols].tolist())
            trainsample.loc[indices[i],'sol'] = p
        else:
            trainsample.loc[indices[i],'sol'] = -1
        if i%100==0:
            print i
    trainsample.loc[indices[rows-1],'sol'] = -1

def getTrain(trainsample):
    userids = trainsample.ncodpers.unique().tolist()
    users = int(len(userids))
    print('userids count:',len(userids))
    trainsample['sol'] = -1
    for i in range(users):
        userid = userids[i]
        trainuser = trainsample[trainsample.ncodpers==userid]
        userindex = trainsample.index[trainsample.ncodpers==userid].tolist()
        getTrainuser(trainuser)
        trainsample.loc[userindex,'sol'] = trainuser.sol.tolist()

def strToOneHot(trainresult):
    strcols = trainresult.select_dtypes(include=['object']).columns.to_list()
    print(strcols)
    for col in strcols:
        trainresult = pd.concat([trainresult,pd.get_dummies(trainresult[col],prefix=col)],axis=1)
    trainresult = trainresult.drop(columns=strcols)
    return trainresult

def strToCatCode(trainresult):
    strcols = trainresult.select_dtypes(include=['object']).columns.to_list()
    for col in strcols:
        trainresult[col] = pd.Categorical(trainresult[col]).codes
    return trainresult

def strToCatCode2(trainresult, cols):
    for col in cols:
        trainresult[col].replace(Dict[col], inplace=True)
    trainresult = trainresult.drop(columns=['ind_empleado', 'pais_residencia', 'canal_entrada'])
    return trainresult

def strRemove(trainresult):
    strcols = trainresult.select_dtypes(include=['object']).columns.to_list()
    trainresult = trainresult.drop(columns=strcols)
    return trainresult

def getTrainSample(train, trainsamplefile, nusers):
    userids = train.ncodpers.unique().tolist()
    usersampleids = userids[:nusers]
    trainsample = pd.DataFrame(columns=train.columns)  
    for i in range(nusers):
        trainsample = trainsample.append(train[train.ncodpers==usersampleids[i]])
    trainsample.to_csv(path_or_buf=trainsamplefile,index=False)

def normalized(traindfcol):
    return (traindfcol - traindfcol.min())/(traindfcol.max() - traindfcol.min()) 

def getTrainFeatures(trainsample, trainfeaturefile):
    remcols = trainsample.filter(regex="ind_.*_ult1").columns.tolist()
    print('get train')
    trainsample = trainsample[trainsample.sol != -1]
    trainsample = trainsample[trainsample.sol.notnull()]
    trainresult = trainsample
    trainresult = trainresult.drop(columns=['ncodpers','fecha_alta','fecha_dato','ult_fec_cli_1t'])
    trainresult[remcols] = trainresult[remcols].fillna(0).astype(int)
    if trainresult.age.dtype in [np.str, np.object]:
        trainresult['age'] = trainresult['age'].str.strip()
        trainresult['age'] = trainresult['age'].str.replace('NA','40',case=True)
    trainresult['age'] = trainresult['age'].fillna('40').astype(int)
    trainresult['age'] = normalized(trainresult['age'])
    if trainresult.antiguedad.dtype in [np.str, np.object]:
        trainresult['antiguedad'] = trainresult['antiguedad'].str.strip()
        trainresult['antiguedad'] = trainresult['antiguedad'].str.replace('NA','0',case=True)
    trainresult['antiguedad'] = trainresult['antiguedad'].fillna('0').astype(int)
    trainresult['antiguedad'] = normalized(trainresult['antiguedad'])
    if trainresult.renta.dtype in [np.str, np.object]:
        trainresult['renta'] = trainresult['renta'].str.strip()
        trainresult['renta'] = trainresult['renta'].str.replace('NA','100000',case=True)   
    trainresult['renta'] = trainresult['renta'].fillna('100000.0').astype(float)
    trainresult['renta'] = normalized(trainresult['renta'])
    trainresult['indrel_1mes'] = trainresult['indrel_1mes'].str.strip()
    trainresult = trainresult.reset_index(drop=True)
    strcols = trainresult.select_dtypes(include=['object']).columns.to_list()
    createDict(trainresult, strcols)
    trainresult = strToCatCode2(trainresult, strcols)
    trainresult.to_csv(path_or_buf=trainfeaturefile,index=False)

#training using xgboost
def xgbTrain(trainfeaturefile):
    trainFeatureData = pd.read_csv(trainfeaturefile)
    remcols = trainFeatureData.filter(regex="ind_.*_ult1").columns.tolist()
    trainLabel = trainFeatureData['sol']
    trainFeatureData = trainFeatureData.drop(columns='sol')
    objectcols = trainFeatureData.select_dtypes(include=['object']).columns.to_list()
    trainFeatureData = trainFeatureData.drop(columns=objectcols)
    print('training')
    mod = xgb.XGBClassifier(objective="multi:softprob", random_state=42, num_class=len(remcols),  subsample=0.7, colsample_bytree=0.7, max_depth=8)
    mod.fit(trainFeatureData,trainLabel, verbose=True, eval_set=[(trainFeatureData, trainLabel)])
    pickle.dump(mod, open(trainfeaturefile+'model.dat',"wb"))
    return mod


#getting test features from train
def generateTestFeatures(train, test, nusers, testfeaturefile):
    print('generating test')
    remcols = train.filter(regex="ind_.*_ult1").columns.tolist()
    testusers = test.ncodpers.sort_values()
    testsample = testusers.iloc[:nusers]
    test2 = test[test.ncodpers.isin(testsample)]
    temp = train[(train.ncodpers.isin(testsample)) & (train.fecha_dato=='2016-05-28')]
    test_X = pd.merge(test2,temp[['ncodpers']+remcols],on='ncodpers')
    test_X = test_X.sort_values(by=['ncodpers'])
    test_X = test_X.drop(columns=['fecha_dato','ncodpers','fecha_alta','ult_fec_cli_1t'])#,'conyuemp'])
    test_X.to_csv(path_or_buf=testfeaturefile,index=False)
    print('cleaning test')
    test_X = pd.read_csv(testfeaturefile)
    if test_X.renta.dtype in [np.str, np.object]:
        test_X['renta'] = test_X['renta'].str.strip()
        test_X['renta'] = test_X.renta.str.replace('NA','100000.00', case=False)
    test_X['renta'] = test_X['renta'].fillna('100000.00').astype(float).astype(int)
    test_X['renta'] = normalized(test_X['renta'])
    if test_X.age.dtype in [np.str, np.object]:
        test_X['age'] = test_X['age'].str.strip()
        test_X['age'] = test_X.age.str.replace('NA','40', case=False)
    test_X['age'] = test_X['age'].fillna('40').astype(int)
    test_X['age']  = normalized(test_X['age'])
    if test_X.antiguedad.dtype in [np.str, np.object]:
        test_X['antiguedad'] = test_X['antiguedad'].str.strip()
        test_X['antiguedad'] = test_X['antiguedad'].str.replace('NA','0',case=False)
    test_X['antiguedad'] = test_X['antiguedad'].fillna('0').astype(int)
    test_X['antiguedad'] = normalized(test_X['antiguedad'])
    test_X['indrel_1mes'] = test_X['indrel_1mes'].astype(str)
    test_X['indrel_1mes'] = test_X['indrel_1mes'].str.strip()
    test_X[remcols] = test_X[remcols].astype(int)
    strcols = test_X.select_dtypes(include=['object']).columns.to_list()
    test_X = strToCatCode2(test_X,strcols)
    test_X = test_X.reset_index(drop=True)
    test_X.to_csv(path_or_buf=testfeaturefile,index=False)

# predicting
def predictTest(testfeaturefile, test, nusers, mod, submissionfile):
    print('predicting')
    test_X = pd.read_csv(testfeaturefile)
    #test_X = test_X.drop(columns=['indrel_1mes'])
    remcols = test_X.filter(regex="ind_.*_ult1").columns.tolist()
    testusers = test.ncodpers.sort_values()
    testsample = testusers.iloc[:nusers]
    prediction = mod.predict(test_X).astype(int)
    submission = pd.DataFrame(columns=['ncodpers','added_products'])
    remcolsdf = pd.DataFrame({'remcols': remcols})
    print('prediction')
    print prediction
    print 'remcolsdf'
    print remcolsdf
    submission['added_products'] = remcolsdf.iloc[prediction].remcols.tolist()
    submission['ncodpers'] = testsample.tolist()
    submission.to_csv(path_or_buf=submissionfile,index=False)
    #display(submission)

def runXGB(train_X, train_y, seed_val=0):
	param = {}
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.05
	param['max_depth'] = 8
	param['silent'] = 1
	param['num_class'] = 24
	param['eval_metric'] = "mlogloss"
	param['min_child_weight'] = 1
	param['subsample'] = 0.7
	param['colsample_bytree'] = 0.7
	param['seed'] = seed_val
	num_rounds = 50

	plst = list(param.items())
	xgtrain = xgb.DMatrix(train_X, label=train_y)
	model = xgb.train(plst, xgtrain, num_rounds)	
	return model

def xgbTrain2(trainfeaturefile):
    trainFeatureData = pd.read_csv(trainfeaturefile)
    remcols = trainFeatureData.filter(regex="ind_.*_ult1").columns.tolist()
    trainLabel = trainFeatureData['sol']
    trainFeatureData = trainFeatureData.drop(columns='sol')
    objectcols = trainFeatureData.select_dtypes(include=['object']).columns.to_list()
    trainFeatureData = trainFeatureData.drop(columns=objectcols)
    print('training')
    train_X = trainFeatureData.to_numpy() 
    train_Y = trainLabel.to_numpy()
    model = runXGB(train_X, train_Y, seed_val=0)
    pickle.dump(model, open(trainfeaturefile+'model.dat',"wb"))
    return model

def predictTest2(testfeaturefile, test, nusers, mod, submissionfile):
    print('predicting')
    test_X = pd.read_csv(testfeaturefile)
    #test_X = test_X.drop(columns=['indrel_1mes'])
    remcols = test_X.filter(regex="ind_.*_ult1").columns.tolist()
    testusers = test.ncodpers.sort_values()
    testsample = testusers.iloc[:nusers]
    xgtest = xgb.DMatrix(test_X.to_numpy())
    prediction = mod.predict(xgtest)
    submission = pd.DataFrame(columns=['ncodpers','added_products'])
    remcolsdf = pd.DataFrame({'remcols': remcols})
    print('prediction')
    print prediction
    print 'remcolsdf'
    print remcolsdf
    prediction = np.argsort(prediction, axis=1)
    prediction = np.fliplr(prediction)[:,:7]
    print('prediction no.')
    print('remcols', remcols)
    print(prediction[0])
    remcols = np.asarray(remcols)
    print(remcols[prediction[0]])
    final_preds = [" ".join(list(remcols[pred])) for pred in prediction]
    print final_preds
    submission['added_products'] = final_preds
    #remcolsdf.iloc[prediction].remcols.tolist()
    submission['ncodpers'] = testsample.tolist()
    submission.to_csv(path_or_buf=submissionfile,index=False)

def getAccuracy(submissionfile, sample_sub, nusers):
    print('result')
    submission = pd.read_csv(submissionfile)
    result = submission['added_products'] == sample_sub[:nusers]['added_products']
    accuracy = 100* result[result==True].shape[0]/nusers
    print(accuracy,'%')

def main():
    train = loadtrain('train_ver2.csv')
    ############trainsamplefile = 'trainsamplefile.csv'
    #############ntrainusers = 1000
    ntrainsamples = len(train.index)#2000000
    ##############getTrainSample(train, trainsamplefile, ntrainusers)
    trainsample = train.sort_values(by=['ncodpers']).iloc[:ntrainsamples]
    
    getTrain2(trainsample)
    ###############trainsample = loadtrain(trainsamplefile)
    trainfeaturefile = 'featureData2-3-full.csv' 
    trainsample.to_csv(path_or_buf=trainfeaturefile+'beforclean.csv', index=False)

    trainsample = pd.read_csv(trainfeaturefile+'beforclean.csv')

    getTrainFeatures(trainsample, trainfeaturefile)

 
    test = loadtest('test_ver2.csv')
    sample_sub = loadsamplesub('sample_submission.csv')    

    testfeaturefile = 'testData2-3.full.csv'
    ntestusers = len(test.index)
    #generateTestFeatures(train, test, ntestusers, testfeaturefile)

    submissionfile = 'submission-3-xbgarr.full-trainfull.csv'
###########################
#    model = xgbTrain(trainfeaturefile)
#    model = pickle.load(open(trainfeaturefile+'model.dat',"rb"))
#    predictTest(testfeaturefile, test, ntestusers, model, submissionfile)
###########################
    model = xgbTrain2(trainfeaturefile)
    model = pickle.load(open(trainfeaturefile+'model.dat',"rb"))
    predictTest2(testfeaturefile, test, ntestusers, model, submissionfile)
    getAccuracy(submissionfile, sample_sub, ntestusers)


if __name__ == '__main__':
    main()

