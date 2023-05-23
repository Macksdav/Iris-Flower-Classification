# Iris-Flower-Classification

Build a Streamlit app that predicts the type of iris flower based on user input using a Random Forest Classifier.

Instructions
Import the necessary libraries: Streamlit, sklearn.datasets, and sklearn.ensemble.
Load the iris dataset using the "datasets.load_iris()" function and assign the data and target variables to "X" and "Y", respectively.
Set up a Random Forest Classifier and fit the model using the "RandomForestClassifier()" and "fit()" functions.
Create a Streamlit app using the "streamlit.title()" and "streamlit.header()" functions to add a title and header to the app.
Add input fields for sepal length, sepal width, petal length, and petal width using the "streamlit.slider()" function. Use the minimum, maximum, and mean values of each feature as the arguments for the function.
Define a prediction button using the "streamlit.button()" function that takes in the input values and uses the classifier to predict the type of iris flower.
Use the "streamlit.write()" function to display the predicted type of iris flower on the app.
Deploy your streamlit app with streamlit share
Note: Make sure to run the app using the "streamlit run" command in your terminal.

 
