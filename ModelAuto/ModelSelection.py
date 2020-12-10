import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#                    Regression Model Selection



def Select_Model_Regression(x_train,x_test,y_train,y_test,degree=2):
    
  
    """[summary]

        DESCRIPTION :-
            This Model will compare all the different Regression models, and will return model with highest Rsq value.
            
            It also shows performance graph comaring the models.

        PARAMETERS :-
            x_train,x_test,y_train,y_test = are the data after tain test split
            
            degree = degree of polinomial regresoin (default = 2)
            
        
        
        Returns:
            Model with heighest Rsq.
            Along with model compaing plot.
            
    """
  

    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    print('\nLinear Regression ...')
    
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    y_pred_lir = lr.predict(x_test)
    lr_pred=r2_score(y_test, y_pred_lir)
    print('Rsq :',lr_pred )

    print('\nPolinomial Regression ...')
    
    polr=PolynomialFeatures(degree)
    x_polr=polr.fit_transform(x_train)
    polr.fit(x_polr,y_train)
    lr.fit(x_polr,y_train)
    y_pred_poly=lr.predict(polr.fit_transform(x_test))
    poly_pred=r2_score(y_pred_poly,y_test)
    print('Rsq :',poly_pred )

    print('\nSVM Model ...')

    regressor = SVR(kernel = 'rbf')
    regressor.fit(x_train, y_train)
    y_pred=regressor.predict(x_test)
    svr_pred=r2_score(y_test,y_pred)
    print('Rsq :',svr_pred)

    print('\nDesision Tree ...')
    
    d_tree=DecisionTreeRegressor(random_state=1)
    d_tree.fit(x_train,y_train)
    y_pred=d_tree.predict(x_test)
    d_tree_acc=r2_score(y_test,y_pred)
    print('Rsq : ',d_tree_acc)

    print('\nRandom Forest ...')
    
    rand = RandomForestRegressor(n_estimators = 100, random_state = 1)
    rand.fit(x_train,y_train)
    y_pred=rand.predict(x_test)
    ran_for_acc=r2_score(y_test,y_pred)
    print('Rsq :',ran_for_acc)

    l=[lr_pred,poly_pred,svr_pred,d_tree_acc,ran_for_acc]
    x_label=['Lin_Reg','Poly_Reg','Svm','Des_Tr','Rand_For']
    ma=l.index(max(l))

    if ma==0:
        model=lr
    elif(ma==1):
        model=polr
    elif(ma==2):
        model=regressor
    elif(ma==3):
        model=d_tree
    else:
        model=rand
        
    xx=np.arange(0,5)
    plt.plot(xx,l)
    plt.ylabel('Rsq')
    plt. xticks(xx,x_label)
    plt.show()

    return model





#                   Classification Model Selection






def Select_model_Classification(x_train,x_test,y_train,y_test):
    
    """[summary]

        DESCRIPTION :-
            This Model will compare all the different Classification models, and will return model with highest Accuracy value.
            
            It also shows performance graph comaring the models.

        PARAMETERS :-
            x_train,x_test,y_train,y_test = are the data after tain test split
        
        Returns:
            Model with heighest Accuracy.
            Along with model compaing plot. 
    """


    from sklearn.linear_model import LogisticRegression 
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    print('LOGISTIC REGRESSIN...\n')

    classifier_log =LogisticRegression(C=1,random_state=0)
    classifier_log.fit(x_train,y_train)
    y_pred = classifier_log.predict(x_test)
    y_pred_log=accuracy_score(y_test, y_pred)
    print('ACCURACY : ',y_pred_log )

    print('\nKNN...\n')

    classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier_knn.fit(x_train, y_train)
    y_pred = classifier_knn.predict(x_test)
    y_pred_knn=accuracy_score(y_test, y_pred)
    print('ACCURACY : ',y_pred_knn )

    print('\nSVM_LINEAR...\n')

    regressor_svmlinear = SVC(kernel = 'linear')
    regressor_svmlinear.fit(x_train, y_train)
    y_pred= regressor_svmlinear.predict(x_test)
    y_pred_svmlin=accuracy_score(y_test,y_pred)
    print('ACCURACY : ',y_pred_svmlin)

    print('\nSVM_NonLinear...\n')

    regressor_svmnon = SVC(kernel = 'rbf')
    regressor_svmnon.fit(x_train, y_train)
    y_pred=regressor_svmnon.predict(x_test)
    y_pred_svmnon=accuracy_score(y_test,y_pred)
    print('ACCURACY : ',y_pred_svmnon)


    print('\nDecision Tree...\n')

    d_tree=DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    d_tree.fit(x_train,y_train)
    y_pred=d_tree.predict(x_test)
    y_pred_dt=accuracy_score(y_test,y_pred)
    print('ACCURACY : ',y_pred_dt)

    print('\nRANDOM FOREST...\n')

    regressor_rf = RandomForestClassifier(n_estimators = 50,random_state = 0,criterion = 'entropy')
    regressor_rf.fit(x_train,y_train)
    y_pred=regressor_rf.predict(x_test)
    y_pred_rf=accuracy_score(y_test,y_pred)
    print('ACCURACY : ',y_pred_rf)

    l=[y_pred_log,y_pred_knn,y_pred_svmlin,y_pred_svmnon,y_pred_dt,y_pred_rf]
    xx=np.arange(0,6)
    plt.plot(xx,l)
    ma=l.index(max(l))
    x_label=['Log_Reg','KNN','Svm_Lin','Svm_Nonlin','RandF','DeciTree']
    plt.ylabel('Accuracy')
    plt. xticks(xx,x_label)
    plt.show()

    if ma==0:
        model=classifier_log
    elif(ma==1):
        model=classifier_knn
    elif(ma==2):
        model=regressor_svmlinear
    elif(ma==3):
        model=regressor_svmnon
    elif(ma==4):
        model=d_tree
    else:
        model=regressor_rf

        return model

