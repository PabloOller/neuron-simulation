import streamlit as st
import numpy as np
import math




style = f"""
<style>
    .appview-container .main .block-container{{
        max-width: 90%;
    }}
</style>
"""
st.markdown(style, unsafe_allow_html=True)




st.image("images/neurona.jpg", width=360)
st.header("Simulador de neurona 4")




class Neuron:

  def __init__(self, weights=[], bias=0, func="sigmoid"):
    self.weights = weights
    self.bias = bias
    self.func = func
  
  def run(self, input_data=[]):
    sum = np.dot(np.array(input_data), self.weights)
    sum += self.bias 

    if self.func == "sigmoid":
      return self.__sigmoid(sum)
    elif self.func == "relu":
      return Neuron.__relu(sum)
    elif self.func == "tanh":
      return self.__tanh(sum)
    else:
      print("La función de activación no está bien.")
      print("funciones de activación permitidas: sigmoid, relu, tanh.")
  
  def changeWeights(self, weights):
    self.weights = weights

  def changeBias(self, bias):
    self.bias = bias
  
  @staticmethod
  def __sigmoid(x):
    return 1 / (1 + math.e ** -x)
  
  @staticmethod
  def __relu(x):
    return 0 if x < 0 else x
  
  @staticmethod
  def __tanh(x):
    return math.tanh(x)




number_of_inputs = st.slider("Nº entradas/sesgos", 1, 10)




st.subheader("Pesos")


w = []
col_w = st.columns(number_of_inputs)

for i in range(number_of_inputs):
    w.append(i)

    with col_w[i]:
        st.markdown(f"w<sub>{i}</sub>", unsafe_allow_html=True)
        w[i] = st.number_input(
            f"w_input_{i}",
            #value=(0.0 if w_option == 'ceros' else round(random() * 10, 2)),
            label_visibility="collapsed")

st.text(f"w = {w}")




st.subheader("Entradas")

x = []
col_x = st.columns(number_of_inputs)

for i in range(number_of_inputs):
    x.append(i)

    with col_x[i]:
        st.markdown(f"x<sub>{i}</sub>", unsafe_allow_html=True)
        x[i] = st.number_input(
            f"x_input_{i}",
            label_visibility="collapsed")

st.text(f"x = {x}")




col1, col2 = st.columns(2)

with col1:
    st.subheader("Sesgo")
    b = st.number_input("Introduce el valor del sesgo")

with col2:
    st.subheader("Función de activación")
    funcion = st.selectbox(
        'Elige la función de activación',
        ('Sigmoide', 'ReLU', 'Tangente hiperbólica'))




FUNCTIONS = {'Sigmoide': 'sigmoid', 'ReLU': 'relu', 'Tangente hiperbólica': 'tanh'}

if st.button("Calcular la salida"):
    n1 = Neuron(weights=w, bias=b, func=FUNCTIONS[funcion])
    st.text(f"Resultado: {n1.run(input_data=x)}")


st.write("Pablo Oller Pérez")