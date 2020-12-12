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
    * [**Demo Code - (Kaggle Housing Price)**](#Demo%Code)

* ### [**Demo on Kaggle - Housing Prices (Advance Regression)**](#Demo%on%Kaggle%-%Housing%Prices)
    
* ### [Conclusion](#Conclusion)
   

## **Usages**
<br>

* ### Getting Started

    ~~~python
        pip install ...
    ~~~
<br>

* ### Demo Code for complete model

    ~~~python


        DATA = '''Importing Data'''
        
                            ''' Data Preprocessing '''

        X_train , X_test = Preprocessing( X_train, X_test , Multi =True)

                            ''' Select best Features '''

        Features = backwardElimination( x_train , y_train )
        
                            '''  Select best Model    '''

        x_train, x_test, y_train, y_test = train_test_split( Features , y_train, test_size=0.2, random_state=1 )

        Model = Select_Model_Regression( x_train,x_test,y_train,y_test )


        Predictions = Model.predict('Test Data')

        print('Done !!!, Now you can make predications !!!😀😀')



                                                     OR

                '''For more choices flexibility data preprocessing could be broken into few quick steps '''


        x_data = handel_nan(x_data)

        y_data = handel_nan(y_data)

        x_train, x_test = handel_Standerd( x_train , x_test )

        x_train , x_test = handel_Catagorical( x_train , x_test )

        
        print("Done !!!😃😃")

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
