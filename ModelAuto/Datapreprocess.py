
# Handling Nan values

import pandas as pd

def handel_nan(DATA, Median=False):
  
    """[summary]

      1. If no. of nan > 50% in a column it will remove that column.
      2. int type coumn will be filled with Mean by Default.
      3. Catagorical columns will be filled by mode.

    Returns:

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


def handel_standardization(train, test=None):
    """[summary]

    ----------------------------------------------------------------

    Example:

      stand_data = handel_standardization( DATA )

                    OR

      X_train ,X_test = handel_standardization (X_data , X_test)

    ----------------------------------------------------------------

    Returns:

        If Input = Train ,Test
        ==> return Train and Test after Standardizing

        If Input = single Dataset
        ==> return Dataset after Standardizing


    """

    if type(test) != type(None):
        data = train.copy()
        Test = test.copy()

    else:
        data = train.copy()

    Row = []

    for row in data:

        if(data[row].dtype != 'O'):
            Row.append(row)

    if(len(Row) != 0):
      
      
      
      from sklearn.preprocessing import StandardScaler
      sc = StandardScaler()

      if type(test) != type(None):

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

      -------------------------------------------------------------------------------------

      Examples:

        x_train , x_test = handel_Catagorical(X_data , x_test)

                            OR

        x_train , x_test = handel_Catagorical(X_data , x_test, selected = [Index of selected Columns --- OR -- Names of columns  ])

                            OR

        DATA = handel_Catagorical ( X_data)

      -------------------------------------------------------------------------------------

      Parameter:

        Train_X = Data or X_data

        Test_Y = If you have seprate Test data

        selected (list ) = User can selected columns on which to perform One hot encoding
                  list- could contain names of columns
                              or
                        could contain Index too .

        remo_dupli = will remove duplicated columns


    Returns:

    """

  DATA_X = Train_X.copy()

  if (type(Test_Y) != type(None)):
      
      DATA_Y = Test_Y.copy()

      data = pd.concat([DATA_X, DATA_Y], axis=0)
  
  else:

      data = DATA_X

  col = DATA_X.columns

  if type(selected) != type(None):

      column = []

      if (type(selected[0]) == int):
          for index in selected:
              column.append(col[index])

      elif (type(selected[0]) == str):
          column = selected
      
      else:
          raise TypeError('Type Error!!')

  for row in data:

      if type(selected) != type(None):

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

  if type(Test_Y) != type(None):

      Train = data.iloc[:len(Train_X), :]
      Test = data.iloc[len(Train_X):, :]

      return (Train, Test)

  else:
    
      return data




def No_of_Catagorical(DATA,graph=False,text=True):
  
  """[summary]
  
    This is a helpfull Vizvalization Methoud
    
    It will show a graph of Total no. of Catagorical Variables in each columns.

  """
  def A(DATA,graph=False,text=True):
    if(graph==True):
      import seaborn as sns
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
      sns.barplot(x=ROW,y=values)

    df['Features']=FEATURE
    df['No_of_Catagorical']=No_of_Catagorical
    print('\n',df,'\n')
    
    
  try:
    A(DATA=DATA,graph=graph,text=text)
  except:
    import pandas as pd
    import seaborn as sns
    A(DATA=DATA,graph=graph,text=text)