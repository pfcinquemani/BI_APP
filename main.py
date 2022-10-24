import codecs
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as stc

st.title("Please choose an option from the sidebar")

menu = ["Choose an option", "Power BI Finance Analysis Dashboard", "Power BI Dashboard with Geolocalization",
        "Python Natural Languaje Processing", "Python Logistic Regresion Graphic"]

choice = st.sidebar.selectbox("Menu", menu)

if choice == "Power BI Finance Analysis Dashboard":
    st.subheader("Finance Analysis Dashboard")

    pagina = "power_bi1.html"
    file = codecs.open(pagina, 'r')
    page = file.read()
    stc.html(page, width=800, height=500, scrolling=False)

if choice == "Power BI Dashboard with Geolocalization":
    st.subheader("Dashboard with Geolocalization")
    pagina = "power_bi2.html"
    file = codecs.open(pagina, 'r')
    page = file.read()
    stc.html(page, width=800, height=500, scrolling=False)


import pandas as pd

# importamos el dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)  # quoting 3 : ignora las comillas dobles

# limpieza de texto
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')  # descarga las palabras inútiles o stop words

corpus = []
for i in range(0, 1000):  # python llega hasta el 999
    review = re.sub("[^a-zA-z]", ' ', dataset['Review'][i])  # eliminamos aquello que no es texto
    review = review.lower()  # pasamos a minúscula
    review = review.split()  # separamos las oraciones por los espacios y hacemos un array
    ps = PorterStemmer()  # convertir en infinitivo las palabras (elimina la declinación o conjugación)
    review = [ps.stem(word) for word in review if not word in set(
        stopwords.words('english'))]  # solo consideramos las palabras que no están en las stop words
    review = ' '.join(
        review)  # join es para unir, pero hay qu eindicarle que separe por algún caracter. Sería el espacio entre las comillas.
    corpus.append(review)  # agrega en la lista

    # cada palabra va a aparecer en columnas separadas
from sklearn.feature_extraction.text import CountVectorizer  # cuenta la cantidad de veces que aparece una palabra

cv = CountVectorizer(
    max_features=1500)  # con esta función podemos hacer lo mismo que la limpieza que hicimos en el bucle for anterior.

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# """Dividimos el dataset en conjunto de entrenamiento y conjunto de testing"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Para este caso eliminamos el escalado de variables ya que este dataset se encuentra entre 0 y 1


if choice == "Python Natural Languaje Processing":


    st.write("This dataset refers to the commentaries of a restaurant clients. "
             "So, here, we are going to test different algorithms in order to determinates witch one is the best to predict if a certain comment is good or bad. "
             "This area of the machine learning is called Natural Processing Languaje and consist on taking raw text, tranform it and give an analysis like we do with numbers.")

    with st.expander("Complete Raw Dataset"):
        st.dataframe(dataset)

    st.subheader("List of Results: ")
    st.write("Here we show the confusion matrix results and four different measures of accuracy of each algorithm.")

    st.write("Naive Bayes")
    # """Creamos el modelo de clasificación"""
    from sklearn.naive_bayes import GaussianNB

    classifier = GaussianNB()  # no tiene parámetros
    classifier.fit(X_train, y_train)

    # """Predicción de los resultado con el conjunto de testing"""
    y_pred = classifier.predict(X_test)

    # """Elaboramos la matriz de confusion para ver si los resultados obtenidos nos sirven de algo"""
    from sklearn.metrics import confusion_matrix

    cm_NB = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write(cm_NB)

    Accuracy_NB = round((cm_NB[1][1] + cm_NB[0][0]) /
                        (cm_NB[1][1] + cm_NB[0][0] + cm_NB[0][1] + cm_NB[1][0]), 2)
    Precision_NB = round((cm_NB[1][1]) / (cm_NB[1][1] + cm_NB[0][1]), 2)
    Recall_NB = round((cm_NB[1][1]) / (cm_NB[1][1] + cm_NB[1][0]), 2)
    F1_Score_NB = round((2 * Precision_NB * Recall_NB) / (Precision_NB + Recall_NB), 2)

    with col2:
        st.write("Accuracy: ", Accuracy_NB)
        st.write("Precision: ", Precision_NB)
        st.write("Recall: ", Recall_NB)
        st.write("F1 Score: ", F1_Score_NB)

    st.write("Logistic Regresion")
    # """Ajustamos el modelo en el conjunto de entrenamiento"""
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    # """Predicción de los resultado con el conjunto de testing"""
    y_pred = classifier.predict(X_test)

    # """Elaboramos la matriz de confusion para ver si los resultados obtenidos nos sirven de algo"""

    cm_LogReg = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write(cm_LogReg)

    Accuracy_logReg = round((cm_LogReg[1][1] + cm_LogReg[0][0]) /
                            (cm_LogReg[1][1] + cm_LogReg[0][0] + cm_LogReg[0][1] + cm_LogReg[1][0]), 2)
    Precision_logReg = round((cm_LogReg[1][1]) / (cm_LogReg[1][1] + cm_LogReg[0][1]), 2)
    Recall_logReg = round((cm_LogReg[1][1]) / (cm_LogReg[1][1] + cm_LogReg[1][0]), 2)
    F1_Score_logReg = round((2 * Precision_logReg * Recall_logReg) / (Precision_logReg + Recall_logReg), 2)

    with col2:
        st.write("Accuracy: ", Accuracy_logReg)
        st.write("Precision: ", Precision_logReg)
        st.write("Recall: ", Recall_logReg)
        st.write("F1 Score: ", F1_Score_logReg)

    st.write("KNN")
    # """Creamos el modelo de clasificación"""
    from sklearn.neighbors import KNeighborsClassifier

    classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski",
                                      p=2)  # métrica minkowly y p=2 es la distancia euclídea.
    classifier.fit(X_train, y_train)

    # """Predicción de los resultado con el conjunto de testing"""
    y_pred = classifier.predict(X_test)

    # """Elaboramos la matriz de confusion para ver si los resultados obtenidos nos sirven de algo"""
    from sklearn.metrics import confusion_matrix

    cm_KNN = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write(cm_KNN)

    Accuracy_KNN = round((cm_KNN[1][1] + cm_KNN[0][0]) /
                         (cm_KNN[1][1] + cm_KNN[0][0] + cm_KNN[0][1] + cm_KNN[1][0]), 2)
    Precision_KNN = round((cm_KNN[1][1]) / (cm_KNN[1][1] + cm_KNN[0][1]), 2)
    Recall_KNN = round((cm_KNN[1][1]) / (cm_KNN[1][1] + cm_KNN[1][0]), 2)
    F1_Score_KNN = round((2 * Precision_KNN * Recall_KNN) / (Precision_KNN + Recall_KNN), 2)

    with col2:
        st.write("Accuracy: ", Accuracy_KNN)
        st.write("Precision: ", Precision_KNN)
        st.write("Recall: ", Recall_KNN)
        st.write("F1 Score: ", F1_Score_KNN)

    st.write("SVM")
    # """Creamos el modelo de clasificación"""
    from sklearn.svm import SVC

    classifier = SVC(kernel="linear", random_state=0)
    classifier.fit(X_train, y_train)

    # """Predicción de los resultado con el conjunto de testing"""
    y_pred = classifier.predict(X_test)

    # """Elaboramos la matriz de confusion para ver si los resultados obtenidos nos sirven de algo"""
    from sklearn.metrics import confusion_matrix

    cm_SVM = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write(cm_SVM)

    Accuracy_SVM = round((cm_SVM[1][1] + cm_SVM[0][0]) /
                         (cm_SVM[1][1] + cm_SVM[0][0] + cm_SVM[0][1] + cm_SVM[1][0]), 2)
    Precision_SVM = round((cm_SVM[1][1]) / (cm_SVM[1][1] + cm_SVM[0][1]), 2)
    Recall_SVM = round((cm_SVM[1][1]) / (cm_SVM[1][1] + cm_SVM[1][0]), 2)
    F1_Score_SVM = round((2 * Precision_SVM * Recall_SVM) / (Precision_SVM + Recall_SVM), 2)

    with col2:
        st.write("Accuracy: ", Accuracy_SVM)
        st.write("Precision: ", Precision_SVM)
        st.write("Recall: ", Recall_SVM)
        st.write("F1 Score: ", F1_Score_SVM)

    st.write("Kernel_SVM")
    # """Creamos el modelo de clasificación"""
    from sklearn.svm import SVC

    classifier = SVC(kernel="rbf", random_state=0)  # cambiamos a radial base function
    classifier.fit(X_train, y_train)

    # """Predicción de los resultado con el conjunto de testing"""
    y_pred = classifier.predict(X_test)

    # """Elaboramos la matriz de confusion para ver si los resultados obtenidos nos sirven de algo"""
    from sklearn.metrics import confusion_matrix

    cm_KernelSVN = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write(cm_KernelSVN)

    Accuracy_KernelSVM = round((cm_KernelSVN[1][1] + cm_KernelSVN[0][0]) /
                               (cm_KernelSVN[1][1] + cm_KernelSVN[0][0] + cm_KernelSVN[0][1] + cm_KernelSVN[1][0]), 2)
    Precision_KernelSVM = round((cm_KernelSVN[1][1]) / (cm_KernelSVN[1][1] + cm_KernelSVN[0][1]), 2)
    Recall_KernelSVM = round((cm_KernelSVN[1][1]) / (cm_KernelSVN[1][1] + cm_KernelSVN[1][0]), 2)
    F1_Score_KernelSVM = round((2 * Precision_KernelSVM * Recall_KernelSVM) / (Precision_KernelSVM + Recall_KernelSVM),
                               2)

    with col2:
        st.write("Accuracy: ", Accuracy_KernelSVM)
        st.write("Precision: ", Precision_KernelSVM)
        st.write("Recall: ", Recall_KernelSVM)
        st.write("F1 Score: ", F1_Score_KernelSVM)

    st.write("Random Forest")
    from sklearn.ensemble import RandomForestClassifier

    classifier = RandomForestClassifier(n_estimators=10, random_state=0,
                                        criterion="entropy")  # default: criterio gini. Aquí utilizaremos entropía.
    # entropía=0 quiere decir que el grupo es totalmento homogéneo. Por lo cual es capaz de clasificar a los elementos con un 100% de seguridad
    # n_estimators: cantidad de árboles
    classifier.fit(X_train, y_train)

    # """Predicción de los resultado con el conjunto de testing"""
    y_pred = classifier.predict(X_test)

    # """Elaboramos la matriz de confusion para ver si los resultados obtenidos nos sirven de algo"""
    from sklearn.metrics import confusion_matrix

    cm_RandomFor = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write(cm_RandomFor)

    Accuracy_RandomForest = round((cm_RandomFor[1][1] + cm_RandomFor[0][0]) /
                                  (cm_RandomFor[1][1] + cm_RandomFor[0][0] + cm_RandomFor[0][1] + cm_RandomFor[1][0]),
                                  2)
    Precision_RandomForest = round((cm_RandomFor[1][1]) / (cm_RandomFor[1][1] + cm_RandomFor[0][1]), 2)
    Recall_RandomForest = round((cm_RandomFor[1][1]) / (cm_RandomFor[1][1] + cm_RandomFor[1][0]), 2)
    F1_Score_RandomForest = round(
        (2 * Precision_RandomForest * Recall_RandomForest) / (Precision_RandomForest + Recall_RandomForest), 2)

    with col2:
        st.write("Accuracy: ", Accuracy_RandomForest)
        st.write("Precision: ", Precision_RandomForest)
        st.write("Recall: ", Recall_RandomForest)
        st.write("F1 Score: ", F1_Score_RandomForest)

    st.write("Decision Tree")
    from sklearn.tree import DecisionTreeClassifier

    classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
    classifier.fit(X_train, y_train)

    # """Predicción de los resultado con el conjunto de testing"""
    y_pred = classifier.predict(X_test)

    # """Elaboramos la matriz de confusion para ver si los resultados obtenidos nos sirven de algo"""
    cm_DecTree = confusion_matrix(y_test, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.write(cm_DecTree)

    Accuracy_ArbolesDecision = round((cm_DecTree[1][1] + cm_DecTree[0][0]) /
                                     (cm_DecTree[1][1] + cm_DecTree[0][0] + cm_DecTree[0][1] + cm_DecTree[1][0]), 2)
    Precision_ArbolesDecision = round((cm_DecTree[1][1]) / (cm_DecTree[1][1] + cm_DecTree[0][1]), 2)
    Recall_ArbolesDecision = round((cm_DecTree[1][1]) / (cm_DecTree[1][1] + cm_DecTree[1][0]), 2)
    F1_Score_ArbolesDecision = round(
        (2 * Precision_ArbolesDecision * Recall_ArbolesDecision) / (Precision_ArbolesDecision + Recall_ArbolesDecision),
        2)

    with col2:
        st.write("Accuracy: ", Accuracy_ArbolesDecision)
        st.write("Precision: ", Precision_ArbolesDecision)
        st.write("Recall: ", Recall_ArbolesDecision)
        st.write("F1 Score: ", F1_Score_ArbolesDecision)

    # construcción del data frame con los resultados

    st.write("Ranking By F1Score")

    st.write("This ranking is ordered descendent by F1Score. "
             "If we assuming that F1Score is the best way to evaluate our analysis, we have to affirm that Naïve Bayes is the best classification algorithm to predict if a punctual commentary is positive or negative. ")

    resultados = {'Algorithm': ['Naive Bayes', 'Logistic Regresion', 'KNN', 'SVM', 'Kernel SVM', 'Random Forest',
                                'Decision Tree'],
                  'Accuracy': [Accuracy_NB, Accuracy_logReg, Accuracy_KNN, Accuracy_SVM, Accuracy_KernelSVM,
                               Accuracy_RandomForest, Accuracy_ArbolesDecision],
                  'Precision': [Precision_NB, Precision_logReg, Precision_KNN, Precision_SVM, Precision_KernelSVM,
                                Precision_RandomForest, Precision_ArbolesDecision],
                  'Recall': [Recall_NB, Recall_logReg, Recall_KNN, Recall_SVM, Recall_KernelSVM, Recall_RandomForest,
                             Recall_ArbolesDecision],
                  'F1Score': [F1_Score_NB, F1_Score_logReg, F1_Score_KNN, F1_Score_SVM, F1_Score_KernelSVM,
                              F1_Score_RandomForest, F1_Score_ArbolesDecision]}

    df = pd.DataFrame(data=resultados)
    df = df.set_index('Algorithm')
    df = df.sort_values('F1Score', ascending=False)
    st.write(df)

    st.write('F1Score Barplot')
    st.bar_chart(df['F1Score'], width=800, height=300, use_container_width=True)

if choice == "Python Logistic Regresion Graphic":
    import pandas as pd

    st.write("This dataset consist on social networks advertisements. "
             "Each observation is a client, or user, and shows differents catracteristics of each one: user ID, gender, age and estimated salary, and finally the data shows if the user has buyed or not."
             "We are going to determinate if the algorithm can predict correctly if the punctual client is going to buy or not, based on the the carasteristics given.")

    st.subheader("Confusion Matrix")

    social_network = pd.read_csv("Social_Network_Ads.csv")

    X = social_network.iloc[:, [2, 3]].values
    y = social_network.iloc[:, 4].values

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    from sklearn.preprocessing import StandardScaler

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)  # en este se hace el fit y el transform
    X_test = sc_X.transform(X_test)  # en este solo el transform porque el fit ya fue hecho anteriormente

    # no es necesario escalar las "y" ya que solo son ceros y unos, podríamos decir que ya están escaladas

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    from sklearn.metrics import confusion_matrix

    cm_log_reg = confusion_matrix(y_test, y_pred)


    col1, col2 = st.columns(2)

    with col1:
        st.write(cm_log_reg)

    Accuracy_ArbolesDecision = round((cm_log_reg[1][1] + cm_log_reg[0][0]) /
                                     (cm_log_reg[1][1] + cm_log_reg[0][0] + cm_DecTree[0][1] + cm_log_reg[1][0]), 2)
    Precision_ArbolesDecision = round((cm_log_reg[1][1]) / (cm_log_reg[1][1] + cm_log_reg[0][1]), 2)
    Recall_ArbolesDecision = round((cm_log_reg[1][1]) / (cm_log_reg[1][1] + cm_log_reg[1][0]), 2)
    F1_Score_ArbolesDecision = round(
        (2 * Precision_ArbolesDecision * Recall_ArbolesDecision) / (Precision_ArbolesDecision + Recall_ArbolesDecision),
        2)

    with col2:
        st.write("Accuracy: ", Accuracy_ArbolesDecision)
        st.write("Precision: ", Precision_ArbolesDecision)
        st.write("Recall: ", Recall_ArbolesDecision)
        st.write("F1 Score: ", F1_Score_ArbolesDecision)

    # Visualising the Train set results
    from matplotlib.colors import ListedColormap

    st.set_option('deprecation.showPyplotGlobalUse', False) #tuve que agregar esto para que pudiese mostrar el gráfico

    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Logistic Regression (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()

    st.pyplot() #se cambia esta línea por plt.show()

    # Visualising the Test set results
    from matplotlib.colors import ListedColormap
    from bokeh.plotting import figure

    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Logistic Regression (Test set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    st.pyplot()



