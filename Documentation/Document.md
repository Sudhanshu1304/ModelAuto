
# Documentation

Bellow is the detailed implientation of the various modules


## **Description** 
<br>

* ### Demo on kaggle - **Advance housing Prices**

    * ### [About]( #About )

    * ### Data preprocessing
        * [Handling Nan]( #Handling%20Nan)
        * [Handeling Standerdization](#Handling&#32;Standerdization)
        * [**Handeling Catagorical Columns**]( #Handling%Catagorical%Columns ) 
        * [**Preprocessing** ( **All in one😀** )]( #Preprocessing )

    * ### Removing Multicolinier
        * [**Using VIF**](#Using%VIF)
        * [**Using Correlation (Faster)**](#Using%Corrilation%(Faster))

    * ### Feature Selection
        * [Backward Elimination](#Backward%Elimination)
        * [Feature Importance](#Feature%Importance)
        * [Corrilation Methoud](#Corrilation%Methoud)
        * [Univarient Selection](#Univarient%Selection)

    * ### Model Selection
        * [Regression Model Selection](#Regress_model)
        * [Classification Model Selection](#Classi_model)


<br><br>

# **Advance Housing Prices**
<br>


<img src="IMAGES\KAGGLE.png" width="800px">
<br>

* ## **About** 

    The dataset of this Housing Prices contains **a total of 163 columns** 
    
    String -> 96<br>
    Int    -> 61<br>
    id     -> 3<br>
    Other  -> 3<br>


    Looking at the above values we could defintly say that it will require  many hours to get at least a good result.
    
    In general the most difficult part in any Model Preparation is 
    
    1. Features selection ( there are tptal of 163 fearures to choose from in this housing model ).
        For which we generaly have to do the analysis of each column and this will take hours.

    2. Model selection - it becomes difficult many time to choose the write type of regression or classification model to select from.

    Let's begin now

    ~~~python

        '''Import Data '''

        Train=pd.read_csv('/content/train.csv')
        Test=pd.read_csv('/content/test.csv')

        X_train=Train.iloc[:,:-1]
        Y_train=Train.iloc[:,-1]
        X_test=Test.iloc[:,:-1]
        Y_test=Test.iloc[:,-1]

        
    ~~~

* ## **Data Preprocessing**
    <br>

    1. ### **Handling Nan**
    
        <br>


        ~~~python
            
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

            x_train = handel_nan( x_train )

        ~~~
    
    2. ### **Handeling Standerdization**

        <br>

        ~~~python

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

            x_train , X_test = handel_standardization( x_train ,X_test )
        ~~~

    3. ### **Handeling Catagorical Columns**

        <br>

        1. This is a extra feature which is very useful in vizvalizaing how many catagorical variables are there in a perticular column.
        This tells us if we should remove a column with lot of catagorical variables since it can increase featues by a greater amount after One hot encoding.

            <br>

            <img src="IMAGES\Catago.png" width=750px >
            
            <br>

            ~~~python

                """[summary]

                    DESCRIPTION :-
                        This is a helpfull Vizvalization Methoud
                        It will show a graph of Total no. of Catagorical Variables in each columns.
                    
                    PARAMETERS :-
                        DATA = Dataset of features
                        graph = Shows a bar graph of No of Catagorical Variables in each column.
                        SIZE = Tuple for gize of the graph.
                        
                    RETURN :-
                        DataFrame of No of Catagorical Variables in each column. 
                        
                """

                NoOfColumns = No_of_Catagorical( x_train )
            ~~~ 
        

        2. **This will convert all the Object type columns in One hot encodings.**
            
            <br>

            ~~~python

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
                                            
                            remo_dupli = will remove duplicated columns if any

                        Returns :-
                            Updated dateset with One hot encoded Catagorical variables.

                """

                x_train , X_test = handel_Catagorical( x_train , X_test)

            ~~~

    4. ### **Preprocessing** 

        <br>

        ~~~python

            def Preprocessing(X_data, X_test=None ,Multi = False):

                """[summary
                    
                    DESCRIPTION :-

                        This will reduce our work by doing the fundamental Preprocessing steps.

                    PARAMETERS :-
                        X_data = features datadet
                        X_test = test datadet
                        Multi = if true it will remove multicollniearity.   
                
                    Returns:
                        Dateframe after doing data preprocessing.  
                """


            x_train , X_test = Preprocessing(X_train ,X_test,Multi=True )

        ~~~



* ## **Removing Multicolinier**
    <br>

    1. ### **Using VIF**
        <br>

        ~~~python
            def handel_Multico_VIF(DATA,sl=5,con=False):

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

            x_train = handel_Multicollinearity_VIF( x_train )
            
        ~~~

    2. ### **Using Correlation (Faster)**

        ~~~python

            """[summary]
                DESCRIPTION :-
                    This Methoud could be used on large or small both type of dataset
                
                PARAMETERS :-
                    DATA = Pandas DataFrame 
                    sl = Columns with Corr > than 0.7 (default) will be removed   
                
                Returns:
                    Updated DataFrame after removing Multicollniearity
                
            """

            x_train = handel_Multicollinearity_Corr( x_train )

        ~~~


* ## **Feature Selection**

    You just have to call these functions they will return the features based on certain parameters you will give.


    1. ### **Backward Elimination**

        <img src="IMAGES\back.png" width="750px" height="300px">

        First graph shows how the accuracy increasing is *incresing* with the *decrese** in features.
        Secound is showing **increase** in accuracy with the **itteration**. But as we can see after some time the change in accuracy stops. It have selected arround 100 features from 160 features.

        ~~~python
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
                
                    Dataframe with features hsving pvalues less than 0.05 (by default).
                
            """
            Features = backwardElimination( x_train, y_train)

        ~~~

    2. ### **Feature Importance**
        <br>

        <img src="IMAGES\FEATURES.png" height="300px" width="750px">

        

        ~~~python 

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

            Features = Feature_Selection(x_train ,Y_train ,122)

        ~~~

    3. ### **Corrilation Methoud**
        <br>

        This Method has two module.

        ~~~python

            1. def Draw_Corr_map(DATA_X,target_column=None,target_index=-1,heat_map=False)

            2. def Corrilation_selection(DATA_X,target_column=None,target_index=-1,Minimum_Corr=0)
    
        ~~~

        **First module** will show the correlation of features with the target column. This will makes easy to decide the minimum correlation value to select features.
        
        <br>
        
        <img src="IMAGES\CORR1.png" height="300px" width="750px" >
    
        ~~~python

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


            Features = Draw_Corr_map(x_train , Y_train)

        ~~~

        ### **Secound module** will return the selected features depending upon the **correlation value selected**.
        <br>

        ~~~python
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

            Features = Corrilation_selection(x_train, Y_train, Minimum_Corr= 0.09)

        ~~~


    4. ### **Univarient Selection**
        <br>

        <img src="IMAGES\UNIVARIENT.png" height="300px" width="750px">

        ~~~python

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

        ~~~

<br>

* ## **Model Selection**
    <br>

    After Selection almost everyrhing is done .
    Now we just have to split the data and pepare one the Models.
    But there are lots of model out there which one to select this is where it comes in handy.

    <img src ="IMAGES\MODEL.png" height="300px" width="750px" >
    
    <br><br>

    1. ### **Regression Model Selection**
        <br>

        ~~~python

            def Regress_model(x_train,x_test,y_train,y_test,degree=2):

                """[summary]

                    DESCRIPTION :-
                        Regression model selection.
                        This Model will compare all the different Regression models, and will return model with highest Rsq value.
                        It also shows performance graph comaring the models.

                    PARAMETERS :-
                        x_train,x_test,y_train,y_test = are the data after tain test split
                        
                        degree = degree of polinomial regresoin (default = 2)
                        
                    Returns:
                        Model with heighest Rsq.
                        Along with model compaing plot.
                        
                """

            Model = Regress_model(x_train,x_test,y_train,y_test)

        ~~~ 


    2. ### **Classification Model Selection**
        <br>

        ~~~python

            def Classi_model(x_train,x_test,y_train,y_test):

                """[summary]

                    DESCRIPTION :-
                        Classification model selection.
                        This Model will compare all the different Classification models, and will return model with highest Accuracy value.
                        
                        It also shows performance graph comaring the models.

                    PARAMETERS :-
                        x_train,x_test,y_train,y_test = are the data after tain test split
                    
                    Returns:
                        Model with heighest Accuracy.
                        Along with model compaing plot. 
                """

        ~~~
        
