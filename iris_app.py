import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

st.title('IRIS FLOWER CLASSIFICATION')

st.header('Classification Solution')
col1, col2 = st.columns(2)
col1.header('Introduction')
col1.write('Iris flower classification is a very popular machine learning project. The iris dataset contains three classes of flowers, Versicolor, Setosa, Virginica, and each class contains 4 features, ‘Sepal length’, ‘Sepal width’, ‘Petal length’, ‘Petal width’. The aim of the iris flower classification is to predict flowers based on their specific features.')

tab1, tab2, tab3 = col2.tabs(["Versicolor", "Setosa", "Virginica"])
with tab1:
   st.header("Versicolor")
   st.image("https://daylily-phlox.eu/wp-content/uploads/2016/08/Iris-versicolor-1.jpg", width=180)

with tab2:
   st.header("Setosa")
   st.image("https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg", width=180)

with tab3:
   st.header("Virginica")
   st.image("https://wiki.irises.org/pub//Spec/SpecVirginica/ivirginicagiantblue01.jpg", width=180)

st.sidebar.image("pngwing.com (19).png", width=100)
st.sidebar.header('choose your parameters')
# st.sidebar.slider('sepal length (cm)', Dataframe['sepal length (cm)'].min())

col1.subheader('Dataframe')

Dataframe = pd.read_csv('iris.data.csv', header=None)
Dataframe.rename(columns = {0: 'sepal length (cm)', 
                       1: 'sepal width (cm)',
                       2: 'petal length (cm)',
                       3:  'petal width (cm)',
                       4: 'Class'}, inplace=True )
st.table(Dataframe.sample(10))
sepal_length = st.sidebar.slider('sepal length (cm)',4.30,7.90,5.84)
sepal_width = st.sidebar.slider('sepal width (cm)',2.00,4.40, 3.05)
petal_length = st.sidebar.slider('petal length (cm)',1.00,6.90,3.76)
petal_width = st.sidebar.slider('petal width (cm)', 0.10,2.50,1.20)


user_input = {'sepal length (cm)': sepal_length,
            'sepal width (cm)': sepal_width,
            'petal length (cm)': petal_length,
            'petal width (cm)':petal_width}

user_data = pd.DataFrame(user_input, index=[0])


#Split into X and Y 
X = Dataframe.drop('Class', axis=1)
Y = Dataframe['Class']
sns.heatmap(X.corr(),fmt='0.1%', cmap='BuPu', annot= True)
plt.savefig('corr.png')
st.image('corr.png', caption='Check for multicollinearity amongst features')
#Split into Training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size= 0.2,
                                                    random_state=43)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
clf = RandomForestClassifier()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
report = classification_report(y_pred, y_test, output_dict=True)
# report.to_csv('Your Classification Report Name.csv', index= True)
report = pd.DataFrame(report, index=None).transpose()
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm/np.sum(cm), fmt='0.1%', annot=True, cmap='Purples')
plt.savefig('save_as_png.png')



st.subheader('Model & Metrics')
tab1, tab2, tab3 = st.tabs(["Classifier", "Classification Report", "Confusion Matrix"])
with tab1:
   st.header("Random Forest Classifier")
   st.write('The Random forest classifier is an ensemble tree-based machine learning algorithm. The random forest classifier is a set of decision trees from a randomly selected subset of the training set. It aggregates the votes from different decision trees to decide the final class of the test object.')

with tab2:
#    st.header("Classification Report")
   st.dataframe(report)

with tab3:
#    st.header("Confusion Matrix")
   st.image('save_as_png.png', width = 500)

# col1,col2 = st.columns(2)
# col1.header("Classification Report")
# st.dataframe(report)

# col2.header("Confusion Matrix")
# col2.image('save_as_png.png', width = 330)

# st.image('save_as_png.png', width = 350)
# st.dataframe(report)


import joblib
joblib.dump(clf, 'RF_clf.pkl')
# user_data['Class'] = clf.predict(user_data)
if st.sidebar.button('Predict'):
   user_data['Class'] = clf.predict(user_data)
   st.text('User Data')
   st.table(user_data)
   st.sidebar.write(user_data['Class'])


#    for i in user_data['Class']:
#         if user_data['Class'] == 'Iris-versicolor':
#             st.image("https://daylily-phlox.eu/wp-content/uploads/2016/08/Iris-versicolor-1.jpg", width=200)
#         elif user_data['Class'].values == 'Iris-setosa':
#            st.image("https://live.staticflickr.com/65535/51376589362_b92e27ae7a_b.jpg", width=200)
#         elif user_data['Class'].values == 'Iris-virginica':
#            st.image("https://wiki.irises.org/pub//Spec/SpecVirginica/ivirginicagiantblue01.jpg", width=200)