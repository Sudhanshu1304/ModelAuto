# **Model Automation**

<br>

This library  will help in automating the processes of Model making (Regression / Classification) ,In few lines of code. One of the major  parts are **Selection of Importent Features and Model .**
**All the Modules are properly documented ,You can find it [here](https://github.com/Sudhanshu1304/ModelAuto/blob/master/Documentation/Document.md)**

The modules help in -

1. Data Preprocessing
2. Removing Multicolinierity
3. **Selecting Features** 😀
4. **Selecting best Model 😉** (Reg- LR , SVM ,... Classif- SVM ,LogR , ...)

> ### **Most of the modules have  graphical techniques for better decision making at Selecting Features and Models**


<br><br>

## **Description**

* ### Usage

    * [Getting Started](#Getting%Started)
    * [Required Libraries](#Required%Libraries)
    * [**Demo Code - (Kaggle Housing Price)**](#Demo%Code)

* ### [**Demo on Kaggle - Housing Prices (Advance Regression)**](#Demo%on%Kaggle%-%Housing%Prices)
    
* ### [Conclusion](#Conclusion)
   

## **Usages**
<br>

* ### Getting Started

    ~~~python

       ' Model Automation '

        pip install ModelAuto

    ~~~
<br>

* ### Required Libraries

    **1. Sklearn**<br>
    **2. seaborn**<br>
    **3. statsmodels**

<br>

* ### Demo Code for complete model

    ~~~python

        from ModelAuto.Datapreprocess import Preprocessing
        from ModelAuto.FeatureSelection import backwardElimination
        from ModelAuto.ModelSelection import Regress_model

        DATA = pd.read_csv('Path/to/file.csv')
        

                            ''' 1. Data Preprocessing '''

        X_train , X_test = Preprocessing( X_train, X_test , Multi =True)

                            ''' 2. Select best Features '''

        Features = backwardElimination( x_train , y_train )
        
                            ''' 3.  Select best Model    '''

        x_train, x_test, y_train, y_test = train_test_split( Features , y_train, test_size=0.2, random_state=1 )

        Model = Regress_model( x_train,x_test,y_train,y_test )

                            ''' 4. Make Predicatoins '''

        Predictions = Model.predict('Test Data')

        
                                print('Done !!!😀😀')

                                                
                '''For more flexibility in Data Preprocessing  '''


        x_data = handel_nan(x_data)

        y_data = handel_nan(y_data)

        x_train, x_test = handel_Standerd( x_train , x_test )

        x_train , x_test = handel_Catagorical( x_train , x_test )


    ~~~

<br>

## **Demo on KAGGLE - Housing Prices (Advance Regression)**

<br>

<img src="Documentation\IMAGES\KAG1.png" height="280px" width="800px">

<br><br>

### [Link to Kaggle Competion](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

A detailed documentation is given on the various modules.
In this documentation implimentation on the **Kaggle Competition** have been shown.
We will be looking at-

How to use various **Feature selection** methods
and Model selection methods.

### [Link to the Documentation](https://github.com/Sudhanshu1304/ModelAuto/blob/master/Documentation/Document.md)



<br><br>

# **Conclusion**

Basically it

1. Speeds up the Model making process .

2. It gives us a clear understanding of the complete model which helps in looking at the minute changes which can boost 
the model efficiency.

3.  A person with very basic knowledge could also create efficient models.

2. The Vizvalization techniques help in understanding concepts easily.    
