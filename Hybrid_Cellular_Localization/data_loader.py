import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

cell_name = "LTU4315A"
cell_data = pd.read_excel(fr"C:\Users\LENOVOGAMING\Desktop\Bitirme Düzenlenmiş Kodlar\{cell_name}.xlsx")
x_m, y_m = cell_data["x_m"], cell_data["y_m"]

mask = y_m > -700 # target user
user_data = cell_data[cell_data["y_m"] > -700]
x_user = x_m[mask]
y_user = y_m[mask]


def fixData():
    data = pd.read_excel(r"C:\Users\LENOVOGAMING\Desktop\Türk Telekom Yeni Veriler\AllDataWithDistance.xlsx")
    cell_name = "LTU4315A"
    cell_data = data[data["SERVING_CELL_NAME"] == cell_name]
    cell_data.drop(index=1674, inplace=True)  # Wrong data because of its time value
    cell_data.to_excel(f"{cell_name}.xlsx")

def getCell():
    cell_data = pd.read_excel(f"{cell_name}.xlsx")

    plt.scatter(x_m[~mask], y_m[~mask], color="blue"), plt.xlabel("$x_{m}$"), plt.ylabel("$y_{m}$")
    plt.scatter(x_m[mask], y_m[mask], color="red", label="Target User"), plt.xlabel("$x_{m}$"), plt.ylabel("$y_{m}$")
    plt.grid(True), plt.title(f"Cell Name: {cell_name}"), plt.legend()
    plt.show()

def getUser():
    plt.scatter(x_m[mask], y_m[mask], color="red"), plt.xlabel("$x_{m}$"), plt.ylabel("$y_{m}$")
    plt.grid(True), plt.title("Target User Trajectory")
    plt.show()

if __name__ == "__main__":
    #fixData()
    getCell()
    getUser()
