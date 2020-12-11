"""   
    This step should only be performed afer handling Catagorical Variables
      
"""

import numpy as np
import pandas as pd

def Get_VIF(X):

    """[summary]

        PARAMETERS :-        
            X = Pandas DataFrame 
        
        Return :-
            Pandas DataFrame of Features and there VIF values
        
    """
    
    def A(X):

        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                           for i in range(len(X.columns))]

        return vif_data

    try:
        A(X)

    except:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        A(X)



# Removing Multicollniearity Methoud - 1


def handel_Multicollinearity_Corr(DATA,sl=0.7):
    
    """[summary]
        DESCRIPTION :-
            This Methoud could be used on large or small both type of dataset
        
        PARAMETERS :-
            DATA = Pandas DataFrame 
            sl = Columns with Corr > than 0.7 (default) will be removed   
        Returns:
        
            Updated DataFrame after removing Multicollniearity
        
    """
    
    
    x_df=DATA
    corr=x_df.corr()
    columns = np.full((corr.shape[0],), True, dtype=bool)
    
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if abs(corr.iloc[i, j]) >= sl:
                if columns[j]:
                    columns[j] = False

    selected_columns = x_df.columns[columns]
    

    return x_df[selected_columns]





# Removing Multicollniearity Method - 2


def handel_Multicollinearity_VIF(DATA,sl=5,con=False):
    
    """[summary]

        DESCRIPTIONS :-
            This Method is only for Small Datasets                                   
                        
        PARAMETERS :-     
            DATA = Pandas DataFrame 
            SL = This will remove all the columns with VIF greater than 5 (default value)
            con = It will add a Constent column if Not present already. 
        
        RETURN :-
            Updated DataFrame after removing Multicollniearity
    
    """

    def A(DATA,sl,con):

      X_df=DATA

      if sl>1:
        
          if con==True:
              X_df.insert(0,'Constant',1,False)
          vif = pd.DataFrame()
          
          for _ in range(X_df.shape[1]-1):
              head=X_df.columns
              list=[vir(X_df.values, j) for j in range(X_df.shape[1])]
              if max(list)>sl:
                  X_df=X_df.drop(columns=[head[list.index(max(list))]],axis=1)

          vif["VIF Factor"] = list
          vif["features"] = X_df.columns
         
          
          return X_df,vif["features"]

      else:

          print('Value of SL should be grater than 1 !!!')


    try:
      A(DATA,sl,con)

    except:       
        
      from statsmodels.stats.outliers_influence import variance_inflation_factor as vir
      import pandas as pd
      import numpy as np
      A(DATA,sl,con)