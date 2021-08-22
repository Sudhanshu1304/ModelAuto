import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

# Handling Nan values

def handel_nan(DATA, Median=False):
  
    """[summary]

        DESCRIPTION :-
            1. If no. of nan > 50% in a column it will remove that column.
            2. int type coumn will be filled with Mean by Default.
            3. Catagorical columns will be filled by mode.
        
        PARAMETERS :-
            DATA = Dataset DataFrame
            Median = It will fill numeric columns with median value.

        Returns :-
            DataFrame with updated Columns


    """

    data = DATA

    for row in data:

        da = data[row]
        NAN = da.isna().sum()
        if(NAN != 0):

            if((NAN/len(da)) >= 0.5):
                data.drop([row], inplace=True, axis=1)
            else:

                if(da.dtype == 'O'):
                    data[row] = data[row].fillna(da.mode()[0])
                else:

                    if Median != True:
                        data[row] = data[row].fillna(da.mean())
                    else:
                        data[row] = data[row].fillna(da.median())

    return data


# Normalizing or Standardizing DataFrame

def handel_standardization(X_train, X_test = None,scale_range=(0,1)):
  
    """[summary]

        DESCRIPTION :-
            stand_data = handel_standardization( DATA )
                          OR
            X_train ,X_test = handel_standardization (X_data , X_test)

        PARAMETERS :-
            X_train = Data or X_data
            X_test  = If you have seprate Test data you can standerdize both at the same time.
            scale_range = it will scale the data between 0,1 by default
            
        Returns:
            If Input = X_train ,X_test
            ==> return Train and Test after Standardizing

            If Input = single Dataset
            ==> return Dataset after Standardizing

    """

    train = X_train
    test  = X_test    
    if test is not None:
        data = train.copy()
        Test = test.copy()

    else:
        data = train.copy()

    Row = []

    for row in data:

        if(data[row].dtype != 'O'):
            Row.append(row)

    if(len(Row) != 0):
      
      from sklearn.preprocessing import MinMaxScaler 
      sc = MinMaxScaler(feature_range=scale_range)

      if test is not None:

          dat = sc.fit_transform(data[Row])
          Tes = sc.transform(Test[Row])
          data[Row] = dat
          Test[Row] = Tes

          return (data, Test)

      else:

          dat = sc.fit_transform(data[Row])
          data[Row] = dat

          return data



# Handling Catagorical Variables

def handel_Catagorical(Train_X, Test_Y=None, selected=None,remo_dupli=True):
  

  """[summary]

        DESCRIPTION :- 
            x_train , x_test = handel_Catagorical(X_data , x_test)
                                OR
            x_train , x_test = handel_Catagorical(X_data , x_test, selected = [Index of selected Columns --- OR -- Names of columns  ])
                                OR
            DATA = handel_Catagorical ( X_data)

        PARAMETERS :-
        
            Train_X = Data or X_data
            Test_Y = If you have seprate Test data
            selected (list ) = [0,4,5] i.e index OR ['feature1','feature2'] i.e names of columns,
            if None is will process all the Catagorical columns
                            
            remo_dupli = will remove duplicated columns if any

        Returns :-
            Updated dateset with One hot encoded Catagorical variables.

  """

  DATA_X = Train_X.copy()

  if (Test_Y is not None):
      
      DATA_Y = Test_Y.copy()

      data = pd.concat([DATA_X, DATA_Y], axis=0)
  
  else:

      data = DATA_X

  col = DATA_X.columns

  if selected is not None:

      column = []

      if (type(selected[0]) == int):
          for index in selected:
              column.append(col[index])

      elif (type(selected[0]) == str):
          column = selected
      
      else:
          raise TypeError('Type Error!!')

  for row in data:

      if selected is not None:

          if row in column:

              da = data[row]
          else:
              continue
      else:
          da = data[row]

      if (da.dtype == 'O'):
          
          dummy = pd.get_dummies(da)
          dummy=dummy.iloc[:,1:]
          
          data.drop([row], inplace=True, axis=1)

          data = pd.concat([data, dummy], axis=1)

  if remo_dupli==True:
    data = data.loc[:,~data.columns.duplicated()]

  if (Test_Y is not  None):

      Train = data.iloc[:len(Train_X), :]
      Test = data.iloc[len(Train_X):, :]

      return (Train, Test)

  else:
    
      return data





def No_of_Catagorical( DATA, graph = True,  SIZE = None ):
  
  """[summary]

    DESCRIPTION :-
        Somtimes converting all the catagorical columns dif=rectelly into One Hot Encoding (dummies) may lead to exponential
        dimentional expanstion hence this plot is there any feature with large no of catagorical variables 
      
        It will show a graph of Total no. of Catagorical Variables in each columns.
    
    PARAMETERS :-
        DATA = Dataset of features
        graph = Shows a bar graph of No of Catagorical Variables in each column.
        SIZE = Tuple for gize of the graph.
        
    RETURN :-
        DataFrame of No of Catagorical Variables in each column. 
        
  """
  def A(DATA,graph=False):
    if(graph==True):
      values=[]
      ROW=[]

    data=DATA
    df = pd.DataFrame()
    FEATURE=[]
    No_of_Catagorical=[]
    for row in data:
      da=data[row]

      if(da.dtype=='O'):

        if(graph==True):
          values.append(da.value_counts().count())
          ROW.append(row)
        
        FEATURE.append(row)
        No_of_Catagorical.append(da.value_counts().count())

    if(graph==True):
      
      if type(SIZE)!= type(None):
          fig = plt.figure(figsize=SIZE)
      
      plt.xlabel('FEATURES')
      plt.ylabel('No. OF COLUMNS')
      sns.barplot(x=ROW,y=values)

    df['Features']=FEATURE
    df['No_of_Catagorical']=No_of_Catagorical
    return df
    
    
  try:
    return A(DATA=DATA,graph=graph)
  except:
    return A(DATA=DATA,graph=graph)
    
    
    


#        Automating Data Prepocessing


def Preprocessing(X_data, X_test=None ,Multi = False,catagorical_columns=None, corr = 0.7,scalerange=(0,1)):
    
    """[summary
    
        DESCRIPTION :-
            This will reduce our work by doing the fundamental Preprocessing steps.
        
        PARAMETERS :-
            X_data = features datadet
            X_test = test datadet
            Multi = if true it will remove multicollniearity using Corelation by Default corr is 0.7.
            catagorical_columns = [0,4,5] OR ['feature1','feature2'],if None is will process all the Catagorical columns
            scalerange = it will scale the data between 0,1 by default 
    
    Returns:
        Dateframe after doing data preprocessing.
        
        
    """
    print('Data Preprocessing...')
    x_train = X_data.copy()
    x_train = handel_nan( x_train)

    if X_test is not None:
        x_test =X_test.copy()
        x_test  = handel_nan(X_test)
        
        x_train , x_test = handel_standardization(x_train,x_test,scale_range=scalerange)
        x_train , x_test = handel_Catagorical(x_train , x_test,selected=catagorical_columns)

    else:
    
        x_train = handel_standardization(x_train,scale_range=scalerange)
        x_train = handel_Catagorical(x_train,selected=catagorical_columns)
    

    if Multi== True:
        
        from Multicollinearity import handel_Multico_Corr
        
        x_train = handel_Multico_Corr(x_train,sl=corr)

    if X_test is not None:
        print('Done!')
        return x_train , x_test
    else:
        print('Done!')
        return x_train

