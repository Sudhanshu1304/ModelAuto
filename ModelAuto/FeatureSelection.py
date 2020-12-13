import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#                                        Method 1                               


def backwardElimination(X_value,Y_value, sl=0.05,plot=True,con=False):
    
    
  """[summary]

    DESCRIPTION :-
        
        This function will show as two graph which will help us in visualization of
        how  is the accuracy increasing with the removal of attributes.
    
    PARAMETERS :-

        X_values = features datadet
        Y_values = target column
        sl = Maximum Pvalue
        con = Constent column if not added by default y = mx + c . c -> c*X0 (constent column)
        plot = True to show plots (default - True)
        
    Returns :-
    
        It will return all the features with pvalues less than 0.05 (by default).
        
    """
  print('Feature Analysing...')
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


def Feature_Importence(X_data,Y_data,Top_Features,plot=True,SIZE=None):
  
  """[summary]
  
      Description :-
        These methoud gives scores for each independent feature. Heigher the score more is its importance.
        This will show us Graph of Top X (given by user) features .
          
      PARAMETERS :-
          X_data = features datadet
          Y_data = target column
          Top_Features = Count of total top features you want
          SIZE = Tuple for size of Plot
        
      Returns:-
          DataFrame of top X (Top_Features) Seleted features
    
  """
  
  print('Feature Analysing...')
  from sklearn.ensemble import ExtraTreesClassifier
  
  if Top_Features==0:
    
    print('\nZero featues seleted !!!')
    return None
  
  elif (Top_Features > X_data.shape[1]):
    
    print('\nSelected More than features avalable !!!')
    Top_Features=X_data.shape[1]
    
    
  if type(SIZE)!= type(None):
    overide =True
  else:
    overide=False
    
    
  
  X=X_data
  Y=Y_data
  No_of_plot=Top_Features
  
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
        fig=plt.figure(figsize=size)

      
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

  return DATA





#                                         METHOD 3






def Draw_Corr_map(DATA_X,target_column=None,target_index=-1,heat_map=False):
  
  """[summary]
                              
    DESCRIPTION :-
        Using this ferture we can see the correlation of features w.r.t target column
        
    PARAMETERS :-
    
        DATA_X = DATA or X_data
        target_column = In case your target is not in the main dataset you can add it here
        target_index = By default it will assumne last column to be the target .
                  ( You can enter Index of target column )   Or  (It can be Name of the column too. )
        heat_map = if true it will show us Heat map
      
    Returns :-
        Dataframe of Features with there Corelation with the target column.
      
  """

  y_index=target_index
  Y_DATA=target_column
  disp_selection = heat_map
  
  DATA=DATA_X.copy()

  if (type(Y_DATA)!=type(None)):
    DATA=pd.concat([DATA,Y_DATA],axis=1)
  

  corre=DATA.corr()
  ROW=list(DATA.columns)
  row=len(ROW)
  
  
  COLUMNS=DATA.columns

  if disp_selection==True:
    
    
    plt.figure(figsize=(row,row))
    graph=sns.heatmap(corre,annot=True,cmap="RdYlGn")
    graph.plot()
    plt.show()
    
    return
  
  else:
    sns.set()

    if (type(y_index)==int):
      ind=y_index
    elif (type(y_index)==str):
      
      try:
        ind=ROW.index(y_index)
      except:
        print('\nIncorrect Name !! Taken Default value\n')
        ind=-1

    corre=corre.iloc[ind]
    r=len(ROW)//15
    CO=(corre)
    if(r!=0):
      a=10
      plt.figure(figsize=(a+r,8))
    
    elif(len(ROW)>5 and len(ROW)<15):
      plt.figure(figsize=(10,6))
    
    plt.bar(COLUMNS,abs(CO),color='#32a852')
    plt.axhline(y=abs(CO).mean(),color='#8400f7',label='Mean')
    plt.axhline(y=abs(CO).std(),color='#0015fc',label='Std_Div')
    plt.axhline(y=abs(CO).median(),color='#f70000',label='Median')
    
    
    plt.title('Corr w.r.t {}'.format(DATA.columns[ind]))

    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()

    print('''

    Mean       : {}
    Median     : {}
    Sta_Div    : {}
  
    '''.format(abs(CO).mean(),abs(CO).median(),abs(CO).std()))
    D=pd.DataFrame({'Columns':COLUMNS,'CORR':CO})
    D = D.reset_index(drop=True)
    
    return D




def Corrilation_selection(DATA_X,target_column=None,target_index=-1,Minimum_Corr=0):

  
  """[summary] 
  
    DESCRIPTION :- 
        This function will return the features whose correlation is above the Minimum_Corr value      
    
    Parameter :-
        DATA_X = Features dataset
        target_column = If your target column is not in the main dataset you can give it here. 
        target_index = the Name or Index of Target column
        Minimum_Corr = Minimum value of Correlation abouve which the features are to be selected.
             
    Returns :-
        DataFrame of Seleted features
      
  """
  print('Feature Analysing...')
  Thresh=Minimum_Corr
  Y_DATA=target_column
  y_index=target_index
  
  DATA=DATA_X.copy()

  if (type(Y_DATA)!=type(None)):

    DATA=pd.concat([DATA,Y_DATA],axis=1)
  

  columns=list(DATA.columns)
  corre=DATA.corr()
  selected=[]

  if (type(y_index)==int):
      ind=y_index
  elif (type(y_index)==str):
    
    try:
      ind=columns.index(y_index)
    except:
      print('\nIncorrect Name !! Taken Default value\n')
      ind=-1
    

  for i in range(len(columns)):
    if (i!=ind):
      if (abs(corre.iloc[ind][i])>=Thresh):
        selected.append(columns[i])
  
  print('Selected Features : ',selected)
  print('No of Features Selected : ',len(selected))

  return DATA[selected]






#                                       Method 4 





def Univariant_Selection(X_data,Y_data,Top_Features,plot=True,SIZE=None):

  """[summary]

    Decription :-
        Univariant methoud basically uses Stastical methoud to find the Features. There are different Stastical methouds depending upon the dataset. 
        different statistical test like chi2 , f_classif tests etc .
        
    PARAMETERS :-
        X_data = features datadet
        Y_data = target column
        Top_Features = Count of total top features you want
      
    Returns:
        
        DataFrame of Seleted features
        
  """
  print('Feature Analysing...')
  
  if Top_Features==0:
    
    print('\nZero featues seleted !!!')
    return 
  
  elif (Top_Features > X_data.shape[1]):
    
    print('\nSelected More than features avalable !!!')
    Top_Features=X_data.shape[1]
    
  
  
  if type(SIZE)!= type(None):
    overide =True
  else:
    overide=False
  
  
  X_value= X_data
  Y_value = Y_data
  No_of_plot=Top_Features
  
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import f_classif
  
  
  features = SelectKBest(score_func=f_classif, k=Top_Features)
  fit = features.fit(X_value,Y_value)


  
  scores= fit.scores_
  
  if plot==True:

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
        fig=plt.figure(figsize=size)

    top_feature = pd.Series(scores, index=X_value.columns) 
    top_feature.nlargest(Top_Features).plot(kind='barh')
    plt.xlabel('Importance of Feature')
    plt.ylabel('Names of Features')
    plt.show()


  val = scores
  ind = np.argpartition(val, -No_of_plot)[-No_of_plot:]
  DATA=X_value.iloc[:,ind]

  print('''

  Max Score     : {}
  Min Score     : {}
  Average Score : {}
  
  '''.format(max(val),min(val),sum(val)/len(val)))
   
  return DATA


