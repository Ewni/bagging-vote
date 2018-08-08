from __future__ import division
from operator import itemgetter

from numpy import *
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines-10,20))        #prepare matrix to return
    mulMat= zeros((20,20))
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    n=0
    index = 0
    for line in fr.readlines():
        if n<4:
            n=n+1
        elif n<numberOfLines-6:
            line = line.strip()
            listFromLine = line.split()
            returnMat[index,:] = listFromLine[2:22]
            classLabelVector.append(int(listFromLine[0]))
            index += 1
            n+=1
    print(returnMat)
    returntMat=returnMat.T
    mulMat=dot(returntMat,returnMat)
    print(mulMat)
    n=shape(mulMat)
    #print(n)
    mulMat=mulMat.reshape(1,400)
    mulMat = array(mulMat)
    #print(mulMat)
    i=shape(mulMat)
    #print(i)
    return mulMat

def percentage2n(eigVals,percentage):
    sortArray=np.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            print(num)
            return num

def pcatrain(dataMat, percentage=0.99):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    topNfeat= percentage2n(eigVals, percentage)
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    print(lowDDataMat)
    #print(eigVals)
    #print(eigVects)
    #print(redEigVects)
    c=lowDDataMat.shape[1]
    print(c)
    a=shape(lowDDataMat)
    print(a)
    return lowDDataMat,c

def pcatest(dataMat, topNfeat):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet


import os
import numpy as np
from sklearn.feature_selection import SelectKBest
from os import listdir
def eachFile(filepath):
    pathDir =listdir(filepath)      #获取当前路径下的文件名，返回List
    m=len(pathDir)
    trainMat=zeros((m,400))
    laber=[]
    for i in range(m):
        fileNameStr = pathDir[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('-')[0])
        laber.append(classNumStr)
    print(laber)
    n=1
    i=0
    for s in pathDir:
        newDir=os.path.join(filepath,s)     #将文件命加入到当前文件路径后面
        if os.path.isfile(newDir) :         #如果是文件
            if os.path.splitext(newDir)[1]==".pssm":  #判断是否是pssm
                resultvec=file2matrix(newDir)                     #读文件
                trainMat[i,:]=resultvec
                i+=1
            else:
                pass
    savetxt('pssm.txt',trainMat, delimiter='\t')
    print("%d"%i)
    print(laber)
    print(trainMat)
    m=shape(trainMat)
    print(m)
    n+=1
    #trainMat=autoNorm(trainMat)
    #trainMat=pcatrain(trainMat)
    #trainMat = pcatrain(trainMat)
    #cross(trainMat,laber)
    #trainMat = SelectKBest(k=300).fit_transform(trainMat, laber)
    print(shape(trainMat))
    trainMat = StandardScaler().fit_transform(trainMat)
    x_train, x_test, y_train, y_test = train_test_split(trainMat, laber, random_state=1, train_size=0.8)
    #W=fwknn(x_train, x_test, y_train, y_test)
    all = [314, 215, 194, 130, 112, 305, 64, 59, 254, 94, 154, 94, 257, 155, 84, 154]
    TP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    FP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #for i in range(2639):
        #ago, now = leaveoneout2(trainMat, laber, i,W)
        #d = int(ago)
        #b = int(now)
        #if (ago == now):
            #TP[d - 1] += 1
        #for a in range(1, 17):
            #if ((ago != now) & (now == a)):
                #FP[a - 1] += 1
    #print(TP)
    #print(FP)
    #x_train,q=pcatrain(x_train)
    #print(q)
    #x_train=x_train.astype(complex)
    #print(x_train)
    #x_test=pcatest(x_test,q)
    #x_test=x_test.astype(complex)
    #print(y_train)
    #a=shape(x_train)
    #print(a)
    #b=shape(x_test)
    #print(b)
    #x_train=pca(x_train)
    #x_test=pca(x_test)
    #print(pcamat)
    #c=shape(pcamat)
    #print(c)
    #i=2
    #wa=[]
    #while i<16:
        #b=adaboost(x_train,y_train,x_test,y_test,i)
        #i+=2
        #wa.append(b)
    #print(wa)
    #vote(x_train, y_train, x_test, y_test,trainMat, laber,W)
    #baggingknn(x_train, y_train, x_test, y_test,trainMat,laber)
    #adaboost(x_train, y_train, x_test, y_test)
    #svm2(x_train,x_test,y_train,y_test)
    KNN(x_train, x_test, y_train, y_test)
    #classify0(x_test,x_train,y_train,1)

def svm2( x_train, x_test, y_train, y_test):
    clf = svm.SVC(C=0.9, kernel='linear', decision_function_shape='ovr',probability=True)
    #clf = svm.SVC(C=0.9, kernel='rbf', gamma=300, decision_function_shape='ovr')
    clf.fit(x_train, y_train)
    print(clf.score(x_train, y_train))  # 精度
    y_hat = clf.predict(x_train)
    print(y_hat)

    print(clf.score(x_test, y_test))
    y_hat = clf.predict(x_test)
    print(y_hat)
    pro=clf.predict_proba(x_test)
    print(pro)


def makecovmat(trainMat):
    #for i in range(3):
        #for j in range(3):
    Y= cov(trainMat,rowvar=0)
            #print(z)
    #X = [trainMat[0], trainMat[1], trainMat[2]]
    #Y = np.cov(X)
    print(Y)
    a=shape(Y)
    print(Y)
    return Y

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
def KNN(train_data,test_data, train_target,test_target):
    clf=KNeighborsClassifier(weights='distance',n_neighbors=1,algorithm='ball_tree',metric='minkowski',p=3)
    clf.fit(train_data, train_target)
    print(clf)
    test_res = clf.predict(test_data)
    print(test_target)
    print(test_res)
    # 打印预测准确率
    pro=clf.predict_proba(test_data)
    print(pro)
    print(accuracy_score(test_res, test_target))


from sklearn.cross_validation import cross_val_score
def cross(X,y):
    #clf=KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=2)
    clf=svm.SVC(C=0.9, kernel='linear', decision_function_shape='ovr')
    #svm.SVC(C=0.9, kernel='rbf', gamma=300, decision_function_shape='ovr')
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    print(scores)
    print(scores.mean())


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier, GradientBoostingClassifier


def adaboost(trainmat,laber,testmat,testlaber):
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15,max_features=None,min_samples_split=6),
                         algorithm="SAMME",n_estimators=30,
                          learning_rate=0.9)
    clf.fit(trainmat,laber)
    scores1 = clf.score(trainmat,laber)
    result=clf.predict(testmat)
    scores = clf.score(testmat, testlaber)
    a=scores.mean()
    print(result)
    print(scores1)
    print(a)
    return result


def leaveoneout2(trainmat,laber,i):
    tmat=zeros((1,400))
    print(shape(laber))
    llaber=laber[:]
    testmat = trainmat[i]
    tmat[0, :] = testmat
    testlaber=llaber[i]
    trainmat2=delete(trainmat,i,axis=0)
    llaber.pop(i)
    print(tmat)
    a=shape(llaber)
    print(a)
    result=vote2(trainmat2,llaber,tmat,testlaber)
    #result=baggingknn(trainmat2,llaber,tmat,testlaber)
    #result=Wknnclass(trainmat2, llaber, tmat, 1,w)
    #result=adaboost(trainmat2,llaber,tmat,testlaber)
    #result=KNN(trainmat2,tmat,llaber,testlaber)
    #result=svm2(trainmat2,tmat,llaber,testlaber)
    return testlaber,result


def knnclass(dataSet, labels,testmat, k):
    dataSetSize = dataSet.shape[0]
    difmat=(testmat-dataSet)
    #print(difmat)
    #print(shape(difmat))
    sqDiffMat = difmat**2
    #print(sqDiffMat)
    #print(shape(sqDiffMat))
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #print(distances)
    sortedDistIndicies = distances.argsort()
    #print(sortedDistIndicies)
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)
    #print(sortedClassCount[0][0])
    return sortedClassCount[0][0]


def Wknnclass(dataSet, labels,testmat, k,w):
    dataSetSize = dataSet.shape[0]
    difmat=(testmat-dataSet)
    #print(difmat)
    #print(shape(difmat))
    sqDiffMat = difmat**2
    sqDiffMat = sqDiffMat * w
    #sqDiffMat=[M * N for M, N in zip(sqDiffMat, w)]
    #print(sqDiffMat)
    #print(shape(sqDiffMat))
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #print(distances)
    sortedDistIndicies = distances.argsort()
    #print(sortedDistIndicies)
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #print(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)
    #print(sortedClassCount[0][0])
    return sortedClassCount[0][0]


def fwknn(x_train, x_test, y_train, y_test):
    #trainmat=x_train[:]
    #testmat=x_test[:]
    #trainlaber=y_train[:]
    #tlaber=y_test[:]
    result = []
    i = 0
    errorsum = 0
    er2mat=[]
    for eschmat in x_test:   #all
        r = knnclass(x_train, y_train, eschmat, 1)
        result.append(r)
        if (y_test[i] != r):
            errorsum += 1
        i += 1
    print(errorsum)
    for n in range(400):    #every feature
        trainmat = x_train[:]
        testmat = x_test[:]
        trainlaber = y_train[:]
        trainmat2= delete(trainmat, n, axis=1)
        testmat2=  delete(testmat, n, axis=1)
        j=0
        errorsum2 = 0
        for eschmat in testmat2:      #delate
            y = knnclass(trainmat2, trainlaber, eschmat, 1)
            #result.append(r)
            if (y_test[j] != y):
                errorsum2 += 1
            j += 1
        er2mat.append(errorsum2)
        print('1')
        print(er2mat)
        print(shape(er2mat))
    for o in range(400):
        er2mat[o]=(er2mat[o]/errorsum)**2
    print(er2mat)
        #er2mat/errorsum
    usum=np.sum(er2mat)
    w=[]
    for m in range(400):
        wi=er2mat[m]/usum
        w.append(wi)
    print(w)
    est=[]
    eserrorsum=0
    t=0
    for efile in x_test:
        es = Wknnclass(x_train, y_train, efile, 1,w)
        est.append(es)
        if (y_test[t] != es):
            eserrorsum += 1
        t += 1
    print(eserrorsum)
    return w


from sklearn.ensemble import BaggingClassifier
def baggingknn(trainmat,laber,testmat,tlaber,mat,mlaber):
    clf1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),algorithm="SAMME", n_estimators=30,learning_rate=0.9)
    clf=KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=1)
    #clf =svm.SVC(C=500, kernel='rbf', gamma=0.001, decision_function_shape='ovr')
    clfb = BaggingClassifier(base_estimator=clf1
                             , max_samples=1.0, max_features=1.0,n_estimators=20)
    clfb.fit(trainmat,laber)

    #predict = clf.predict(trainmat)
    #result= clfb.predict(testmat)

    #print(clf.score(trainmat,laber))
    #print(clf.score(testmat,tlaber))
    #result2=clfb.score(testmat, tlaber)
    score=clfb.score(testmat, tlaber)
    print(clfb.score(testmat, tlaber))
    #print(result)
    score1c1 = cross_val_score(clf, mat, mlaber, cv=5, scoring='accuracy')
    scorec2 = cross_val_score(clfb, mat, mlaber, cv=5, scoring='accuracy')
    print('knn')
    print(score1c1.mean())
    print('bagging')
    print(scorec2.mean())
    return score
    # print Series(predict).value_counts()
    # print Series(predictb).value_counts()


from sklearn.ensemble import RandomForestClassifier
def vote(trainmat,laber,testmat,llaber,W):
    clf1 = svm.SVC(C=500, kernel='rbf', gamma=0.001, decision_function_shape='ovr')
    clf2 = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=1)
    clf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),
                             algorithm="SAMME", n_estimators=30,
                             learning_rate=0.9)
    clf4 = RandomForestClassifier(min_samples_split=2,n_estimators=140,min_samples_leaf=2, max_depth=14, random_state=60)
    clf5=GradientBoostingClassifier(learning_rate=0.1, n_estimators=150, min_samples_split=2,min_samples_leaf=20, max_depth=16, subsample=0.8, random_state=10)
    #W = fwknn(trainmat, testmat, laber, llaber)
    clf4=Wknnclass(trainmat, laber, testmat, 1, W)

    #eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('tree', clf3),('rand',clf4)],voting='hard')
    #eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('tree', clf3)], voting='hard')
    eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2),  ('rand', clf4)], voting='hard')
    #eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('gbdt', clf5)], voting='hard')
    #eclf1 = VotingClassifier(estimators=[('knn', clf1), ('tree', clf3), ('rand', clf4)], voting='hard')
    #eclf1 = VotingClassifier(estimators=[('svm', clf1), ('knn', clf2), ('tree', clf3)], voting='soft',weights=[0.95,0.96,0.956],flatten_transform=True,)
    eclf1.fit(trainmat, laber)
    result=eclf1.predict(testmat)
    #score=eclf1.score(testmat,llaber)
    #score=eclf1.predict_proba(testmat)
    #clf=clf2.fit(trainmat,laber)
    #print("knn")
    #print(clf.score(testmat,llaber))
    #print("vote")
    #print(score)
    #score1c1=cross_val_score(clf,mat,mlaber,cv=10,scoring='accuracy')
    #scorec2=cross_val_score(eclf1,mat,mlaber,cv=10,scoring='accuracy')
    #print('knn')
    #print(score1c1.mean())
    #print('vote')
    #print(scorec2.mean())
    return result

def vote2(trainmat,laber,testmat,llaber):
    clf1 = svm.SVC(C=500, kernel='rbf', gamma=0.001, decision_function_shape='ovr')
    clf2 = KNeighborsClassifier(weights='distance', n_neighbors=1, algorithm='ball_tree', metric='minkowski', p=1)
    clf3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, max_features=None, min_samples_split=6),
                             algorithm="SAMME", n_estimators=30,
                             learning_rate=0.9)
    clf4 = RandomForestClassifier(n_estimators=120)
    #W = fwknn(trainmat, testmat, laber, llaber)
    #clf4=Wknnclass(trainmat, laber, testmat, 1, W)

    eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('tree', clf3),('rand',clf4)],voting='hard')
    #eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('tree', clf3)], voting='hard')
    eclf1 = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2),  ('rand', clf4)], voting='hard')
    #eclf1 = VotingClassifier(estimators=[('knn', clf1), ('tree', clf3), ('rand', clf4)], voting='hard')
    #eclf1 = VotingClassifier(estimators=[('svm', clf1), ('knn', clf2), ('tree', clf3)], voting='soft',weights=[0.95,0.96,0.956],flatten_transform=True,)
    eclf1.fit(trainmat, laber)
    result=eclf1.predict(testmat)
    #score=eclf1.score(testmat,llaber)
    #score=eclf1.predict_proba(testmat)
    #clf=clf2.fit(trainmat,laber)
    #print("knn")
    #print(clf.score(testmat,llaber))
    #print("vote")
    #print(score)
    return result


eachFile('z')
#file2matrix('11-1.pssm')




