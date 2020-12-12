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
        * Handeling Catagorical Columns. 
        * **Preprocessing** ( **All in one🤔** )

    * Removing Multicolinier
        * Using VIF
        * **Using Correlation (Faster)**

    * Feature Selection
        * Backward Elimination
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

