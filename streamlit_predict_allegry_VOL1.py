#Carregando as bilbiotecas 
import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt
import numpy as np 
import plotly.express as px 
import xlrd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_curve, recall_score

#DATA 
#Data para o algoritmo
data = pd.read_excel('food.xlsx')

#Data para o gráfico de alergias
food_all = pd.read_csv('FoodData.csv')
#UI 
#Título da página 
st.title("Modelo de predição de alergia em crianças")

#Fazendo descrição 
st.write('\n')
st.write("Preencha os parâmetros para utilizar o algoritmo de classificação")
st.write("Veja as caixas abaixo para entender melhor o significado de cada campo")
st.write('\n')

#Criando as caixas de expansão (Técnico)
Tech = st.beta_expander("Téncico")
Tech.write("""Bibliotecas utilizadas:
\n-Pandas
\n-Matplotlib
\n-Scikit-Learn""")

#Criando as caixas de expansão (Dados)
data_descr = st.beta_expander("Dados")
data_descr.write("Os dados utilizados para a construção desse algoritmo foram criados de forma aleatória")

#Criando as caixas de expansão (Dados descrição colunas)
columns_descr = st.beta_expander('inputs')
columns_descr.write("""Como preencher os dados?
Todos os dados irão do número 1 a 5, ou seja, coloque o grau que cada sintoma teve na criança""")

#Criando o sidebar 
st.sidebar.title("Preencha os parâmetros")
st.sidebar.write("Caso haja dúvidas de como preencher veja as caixas de expansão")

#Colocando os parâmetros
espirros_input = st.sidebar.number_input("Espirros", 0, 5)
nariz_input = st.sidebar.number_input("Nariz entupido", 0, 5)
olhos_input = st.sidebar.number_input("Olhos vermelhos", 0, 5)
tosse_input = st.sidebar.number_input("Tosse", 0, 5)
ar_input = st.sidebar.number_input("Falta de ar", 0, 5)
mancha_input = st.sidebar.number_input("Macha no corpo", 0, 5)
dor_input = st.sidebar.number_input("Dor no corpo", 0, 5)
diarreia_input = st.sidebar.number_input("Diarréia", 0, 5)

#Colocando o botão 
button = st.sidebar.button("Fazer o diagnóstico")


#CLASSIFICAÇÃO
#Separando os dados em features e label 
features = ['Espirros','Nariz_entupido','Olhos_vermelhos','Tosse','Falta_ar','Machar_corpo','Dor_corpo','diarreia']
label = ['Resultado']

#Criando as variavéis 
X, y = data[features].values, data[label].values

#Separando os dados para a criação do modelo 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Pegando o modelo 
model = LogisticRegression()

#Treinando o modelo 
model.fit(X_train, y_train)

#Fazendo as predições 
predictions = model.predict(X_test)

##############################################################

#TESTE
Htmlfile = open('teste.html','r', encoding='utf-8')
source_code = Htmlfile.read()
print(source_code)
components.html(source_code, height=1000)

#Pegando os inputs 
X_new = np.array([espirros_input, nariz_input,olhos_input,tosse_input,ar_input,mancha_input,dor_input,diarreia_input])
X_new = X_new.reshape(1,-1)

if button:
    prediction = model.predict(X_new)
    prob = model.predict_proba(X_new)[:,1]
    st.write('\n')
    st.write('\n')
    st.write('\n')
    if prediction == 1:
        st.write('O paciente tem **{}** de ser alérgico ao alimento que consumiu'.format(prob))
    else:
        st.write('O paciente não é alérgico ao alimento que consumiu. Sua chance de ser alérgico é de **{}**'.format(prob))

    #Divindo as colunas 
    c1, c2 = st.beta_columns((2,2))

    #Pegando as métricas 
    c1.write('### As principais métricas do modelo')

    #Fazendo as métricas
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    #Mostrando as métricas 
    c1.write('Acurácia do modelo: **{:0%}**'.format(acc))
    c1.write('Acurácia do modelo: **{:0%}**'.format(prec))
    c1.write('Recall do modelo: **{:0%}**'.format(recall))

    #Mostrando a curva ROC 
    c2.write('### Curva ROC do modelo')

    #Criando a curva ROC

    #Pegando as informações da curva 
    fpr, tpr, threslholds = roc_curve(y_test, predictions)

    #Plotando a curva 
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("Falsos positivos Rate")
    plt.ylabel('Verdadeiros positivos Rate')
    plt.title('ROC Curve')

    #Plotando as linhas 
    plt.plot([0,1],[0,1],"k--")

    #Colocando o gráfico na coluna
    c2.write(fig)

    #Colocando o mapa com os médicos
    st.write("# Encontre um médico Danone perto de você")
    st.write("### Mesmo com a ajuda do nosso algoritmo é necessário um especialista para realizar o melhor diagnóstico para o paciente")
    st.write("### Encontre o médico/pediatra/nutricionista mais perto da sua localização")

    #Colocando o mapa
    st.map()

    #Colocando o gráfico sobre alergia 

    #Texto 
    st.write("# Veja quais são os tipos, classe e comidas e suas respectivas alergias")
    st.write("### A equipe Danone separou uma lista completa de alimentos que podem causar alergia para você ficar de olho naqueles que possam ter alguma reação alergica")

    #Criando o g´rafico 
    fig = px.sunburst(food_all, path=['Class','Type','Group','Food','Allergy'])

    #Plotando o gráfico 
    st.plotly_chart(fig)

