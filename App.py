import os
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,12)

def function(data, a, b, c, d, e, f, g, h):
    x = data[0]
    y = data[1]
    return a + (b * (x**2) * y) + (c * (y**2) * x) + (d * (x**2)) + (e * (y**2)) + (f * x * y) + (g * x) + (h * y)

def function2(data, a, b, c, d, e, f):
    x = data[0]
    y = data[1]
    return a + (b * (x**2)) + (c * (y**2)) + (d * x * y) + (e * x) + (f * y)

def function3(data, a, b, c, d, e, f, g, h, i, j):
    x = data[0]
    y = data[1]
    return a + (b * (x**3)) + (c * (y**3)) +  (d * (x**2) * y) + (e * (y**2) * x) + (f * (x**2)) + (g * (y**2)) +  (h * x * y) + (i * x) + (j * y)

def function4(data, a, b, c):
    x = data[0]
    y = data[1]
    return a + x**b + y**c

def function5(data, a, b, c, d, e):
    x = data[0]
    y = data[1]
    return a + b * (x**c) + d * (y**e)

def function6(data, a, b, c, d, e):
    x = data[0]
    y = data[1]
    return a + x**b + y**c + (x**d) * (y**e)

def function7(data, a, b, c, d, e, f, g, h):
    x = data[0]
    y = data[1]
    return a + b * (x**c) + d * (y**e) + f * (x**g) * (y**h)

def function8(data, a, b, c, d, e, f, g, h, i, j, k, l):
    x = data[0]
    y = data[1]
    return a + (b * (x**c)) + (d * (y**e)) + (f * (x**g) * (y**h)) + (i * (x**j) * y) + (k * x * (y**l))

def function9(data, a, b, c, d, e, f, g, h, i, j):
    x = data[0]
    y = data[1]
    return a + (b * (x**c)) + (d * (y**e)) + (f * (x**g) * (y**h)) + np.log(i * x) + np.log(j * y)

def function10(data, a, b, c, d, e, f, g, h, i):
    x = data[0]
    y = data[1]
    return a + (b * (x**c)) + (d * (y**e)) + (f * (x**g) * (y**h)) + np.log(i * x * y)

def function11(data, a, b, c, d, e, f, g, h, i, j, k):
    x = data[0]
    y = data[1]
    return a + (b * (x**c)) + (d * (y**e)) + (f * (x**g) * (y**h)) + np.log(i * x * y) + np.log(j * x) + np.log(k * y)

def function12(data, a, b, c, d, e, f, g, h, i, j, k, l, m, n):
    x = data[0]
    y = data[1]
    return a + (b * (x**c)) + (d * (y**e)) + (f * (x**g) * (y**h)) + i * np.log(j * x * y) + k * np.log(l * x) + m * np.log(n * y)

def function13(data, a, b, c, d, e, f, g):
    x = data[0]
    y = data[1]
    return a + b * np.log(c * x * y) + d * np.log(e * x) + f * np.log(g * y)

def function14(data, a, b, c):
    x = data[0]
    y = data[1]
    return a + b**x + c**y

def metrics(data, function, *parameters):
    x = data[0]
    y = data[1]
    z = data[2]
    AE = np.abs(z - function(np.array([x, y]), *parameters))
    PE = (AE/z) * 100
    SE = np.square(AE)
    MSE = np.mean(SE)
    RMSE = np.nan_to_num(np.sqrt(MSE))
    # print("MAX METRICS:\nABSOLUTE ERROR:{}\nPERCENTAGE ERROR: {}\nSQUARE ERROR: {}\nMEAN SQUARE ERROR: {}\nROOT MEAN SQUARE ERROR: {}" .format(np.max(AE), np.max(PE), np.max(SE), np.max(MSE), np.max(RMSE)))
    return AE, PE, SE, MSE, RMSE

def model(data, function, mesh_finesse=15000):
    x = data[0]
    y = data[1]
    z = data[2]
    parameters, covariance = curve_fit(function, [x_data, y_data], z_data, maxfev=10000000)
    model_x_data = np.linspace(min(x_data), max(x_data), mesh_finesse)
    model_y_data = np.linspace(min(y_data), max(y_data), mesh_finesse)
    X, Y = np.meshgrid(model_x_data, model_y_data)
    Z = function(np.array([X, Y]), *parameters)
    return X, Y, Z, parameters, covariance

st.title("Test")

col0 = st.beta_container()

col1, col2 = st.beta_columns(2)

filepath = st.file_uploader("File Uploader")
if (filepath != None):

    dataframe = pd.read_csv(filepath)
    
    x_data = dataframe.iloc[:, 0]
    y_data = dataframe.iloc[:, 1]
    z_data = dataframe.iloc[:, 2]

    func = 0

    for x in [function, function2, function3, function4, function5, function6, function7, function8, function9, function10, function11, function12, function13, function14]:
        
        func = func + 1
        
        X, Y, Z, parameters, covariance = model([x_data, y_data, z_data], x, mesh_finesse=2500)

        if np.isnan(parameters).any():
            print("NaN")

        else:
            AE, PE, SE, MSE, RMSE = metrics([x_data, y_data, z_data], x, *parameters)


            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(X, Y, Z)
            ax.scatter(x_data, y_data, z_data, color='red')
            ax.set_xlabel('X data')
            ax.set_ylabel('Y data')
            ax.set_zlabel('Z data')
            ax.view_init(15, 75)
            # ax.text3D(0, 0, 0, "MSE: {}" .format(round(MSE, 2)), color = "red", size = "xx-large")
            ax.text3D(0, 0, 0, "MSE: {}" .format(round(RMSE, 2)), color = "blue", size = "xx-large")
            plt.savefig("graph_"+str(func)+".jpg")
            st.image("graph_"+str(func)+".jpg")
            plt.show()

            with col0:
                st.text(parameters)

                with col1:
                    st.dataframe(dataframe)

                with col2:
                    results = pd.DataFrame([AE, PE, SE], index = ["AE", "PE", "SE"]).T
                    st.dataframe(results)