# Model Automation

This library  will help in automating the process of model making (Regresion / Classification).
It will help in
1. Data Preprocessing
2. Removing Multicolinierity
3. **Selecting Features** ❤
4. Selecting best Model { Regress - (LR ,..,Random Forest) , Classi - (LogR ,..,Svm..) }

**Advance housing :- A small demo has been given bellow** 


Description :-

* Usages
    * Getting Started
    * Sample Code for complete model
* Demo on kaggle - **Advance housing Prices**

    * About

    * Data preprocessing
        * Handling Nan
        * Handeling Standerdization
        * Hamdeling Catagorical Columns. 

    * Removing Multicolinier
        * Using VIF
        * Using Correlation

    * Feature Selection
        * backward Elimination
        * Feature Importance
        * Corrilation Methoud
        * Univarient Selection

    * Model Selection
        * Regression Model Selection
        * Classification Model Selection


## **Usages**

* Getting Started

    ~~~python

        pip install ...

    ~~~

* Sample Code for complete model

    ~~~python

        DATA = '''Import data'''

        ''' Datapreprocessing '''

        x_data = handel_nan(x_data)

        y_data = handel_nan(y_data)

        x_train, x_test = handel_Standerd( x_train , x_test )

        x_train , x_test = handel_Catagorical( x_train , x_test )

        ''' Feature Selection '''

        selected_features = backwardElimination( x_train , y_train )
        
        ''' Model Execution '''

        x_train,x_test,y_train,y_test = train_test_split( selected_features , y_train, test_size=0.2, random_state=1 )

        selected_model = Select_Model_Regression( x_train,x_test,y_train,y_test )

        print(' Model Accurecy is : ', selected_model.predict(x_test) )
        
        print("Done !!!😃😃")

    ~~~


* ## Demo on kaggle - **Advance housing Prices**
<br>

![](ModelAuto\IMAGES\KAGGLE.png)

    * About 

        The dataset of this Housing Prices contains **a total of 163 columns** 
        String -> 96
        Int    -> 61
        id     -> 3
        Other  -> 3

        Looking at the above values we could defintly say that it will require  many hours to get at least a good result.
        
        In general the most difficult part in any Model Preparation is -
        
            1. Features selection ( there are tptal of 163 fearures to choose from in this housing model ).
            For which we generaly have to do the analysis of each column and this will take hours.

            2. Model selection - it becomes difficult many time to choose the write type of regression or classification model to select from.

        Let's begin now

            ~~~python

            ~~~

    * Data Preprocessing

