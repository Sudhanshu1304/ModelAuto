

#                                        Method 1                               


def backwardElimination(X_value,Y_value, sl=0.05,con=False,plot=True):
    
    
    """[summary]
    
    Vizvalization:
    
        This function will show as two graph which will help us in vizvalizing
        how  is the accuracy increasing with the removal of attributes.
        
    Returns:
    
        It will return all the features with pvalues less than 0.05 (by default).
        
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    import statsmodels.regression.linear_model as lmd
  

    X=X_value.copy()
    Y=Y_value

    n_columns=[]
    n_ittere=[]
    y_accu=[]

    if con==True:

        ran=list(np.ones((len(X)),dtype='int'))
        X.insert(0, 'Constent_Row', ran)
    
    x=X
    y=Y

    numVars = len(x.columns)
    r_sq=0
    
    for i in range(0, numVars):

        print('.',end=' ')

        regressor_OLS = lmd.OLS(endog=y, exog=x).fit()

        if regressor_OLS.rsquared_adj>=r_sq:
            
            
            n_columns.append(len(x.columns))
            n_ittere.append(i)
            y_accu.append(regressor_OLS.rsquared_adj)

            r_sq=regressor_OLS.rsquared_adj
            maxVar = float(max(regressor_OLS.pvalues))
            if maxVar > sl:
                for j in range(0, numVars - i):
                    if (float(regressor_OLS.pvalues[j]) == maxVar):
                
                        x = x.drop( x.columns[[j]], axis=1)
                        break

    if(plot==True):
      fig,ax=plt.subplots(nrows=1, ncols=2,figsize=(14,5))

      ax[0].plot(n_columns,y_accu)
      ax[0].set_xlabel('No. of Columns')
      ax[0].set_ylabel('ACCURACY')
      ax[1].plot(n_ittere,y_accu)
      ax[1].set_xlabel('ITERATION')
      ax[1].set_ylabel('ACCURACY')
      plt.show()
      
    return x






#                                      Method 2 


def Feature_Selection(X,Y,No_of_plot,plot=True,overide=False,SIZE=None):
  
  
  
  """[summary]

    These methoud gives scores for each independent feature. Heigher the score more is its importance.
        
    Vizvalization : This will show us Graph of Top X (given of user) features .
        
        
    Returns:
        
        DataFrame of Seleted features
    
    """
  
  
  from sklearn.ensemble import ExtraTreesClassifier
  import numpy as np

  model = ExtraTreesClassifier()
  model.fit(X,Y)

  if(plot==True):

    import matplotlib.pyplot as plt
    
    if No_of_plot>=20 and No_of_plot<30 :
      size=5
      fig=plt.figure(figsize=(size,size))

    elif No_of_plot>=30 and No_of_plot<60:
      size=9
      fig=plt.figure(figsize=(size,size))

    elif No_of_plot>=60 and No_of_plot<90:
      size=13
      fig=plt.figure(figsize=(size,size))

    elif No_of_plot>=90:
      size=15
      fig=plt.figure(figsize=(size,size))
    
    if (overide==True):
      if (SIZE==None):
        print('\nNew Size is not given by the User !!! default taken ')
      else:
        size=SIZE
        fig=plt.figure(figsize=(size,size))

      
    scor=model.feature_importances_
    plt.axvline(x=abs(scor).mean(),color='#f20292',label='Mean')
    plt.axvline(x=abs(scor).std(),color='#15ff00',label='Std_Div')
    plt.axvline(x=np.median(abs(scor)),color='red',label='Median')
    top_feature = pd.Series(scor, index=X.columns)
    top_feature.nlargest(No_of_plot).plot(kind='barh',label='Importance')
    plt.legend()
    plt.xlabel('Importance of Feature')
    plt.ylabel('Names of Features')
    plt.show()
  
 
  val = model.feature_importances_
  ind = np.argpartition(val, -No_of_plot)[-No_of_plot:]
  DATA=X.iloc[:,ind]

  print('''

  Max Score     : {}
  Min Score     : {}
  Average Score : {}
  
  '''.format(max(val),min(val),sum(val)/len(val)))

  return (DATA,val)



