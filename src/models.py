import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.preprocessing as prepro
# import preprocessing as prepro
# import data_handler
# from utils import RawData

class LinealReg:
    def __init__(self, x : np.ndarray, y : np.ndarray, L1 : float = 0, L2 : float = 0, initial_weight_value : float = 1):
        self.x : np.ndarray = np.array(np.c_[np.ones(x.shape[0]), x], dtype=np.float64)   # agrego columna de unos para el bias.
        self.y : np.ndarray = np.array(y, dtype=np.float64)
        self.L1 = L1
        self.L2 = L2
        self.coef : np.ndarray = np.full(shape=self.x.shape[1], fill_value=initial_weight_value)
    
    def fit_pseudo_inverse(self):
        # w = (X^T * X)^-1 * X^T * Y
        # w = ((X^T * X) + L2*Id)^-1 * X^T * Y
        # self.coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.x.T, self.x)), self.x.T), self.y)
        # self.coef = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.x.T, self.x) + (self.L2 * np.identity(self.x.shape[1]))), self.x.T), self.y)
        # self.coef = np.linalg.inv(self.x.T @ self.x) @ self.x.T @ self.y
        self.coef = np.linalg.inv((self.x.T @ self.x) + (self.L2 * np.identity(self.x.shape[1]))) @ self.x.T @ self.y
        # print(self.error_cuadratico_medio())

    def fit_gradient_descent(self, step_size : float, tolerance : float = -1, max_number_of_steps : int = -1):
        attempts = 0
        # print(self.x.shape)
        while True:
            gradient = self.least_squares_gradient()
            if (np.linalg.norm(gradient) <= tolerance and tolerance != -1) or (attempts >= max_number_of_steps and max_number_of_steps != -1):
                break
            # print("(step_size * (gradient / np.linalg.norm(gradient)): ", (step_size * (gradient / np.linalg.norm(gradient))))
            # print("(gradient / np.linalg.norm(gradient): ", (gradient / np.linalg.norm(gradient), "\n"))
            # print(self.error_cuadratico_medio())
            self.coef = self.coef - (step_size * (gradient))
            attempts += 1
            # print(self.error_cuadratico_medio())

    def error_least_squares_function(self) -> float:
        # ||Xw - Y||^2
        return np.linalg.norm((self.x @ self.coef) - self.y)**2

    def error_cuadratico_medio(self, validation_set_y : np.ndarray = None, validation_set_x : np.ndarray = None) -> float:
        val_set_y : np.ndarray = self.y
        val_set_x : np.ndarray = self.x
        if validation_set_y is not None and validation_set_x is not None:
            # print("HOLA")
            val_set_y = validation_set_y
            val_set_x = np.array(np.c_[np.ones(validation_set_x.shape[0]), validation_set_x], dtype=np.float64)
        # print(val_set_x.shape)
        # print(val_set_y.shape)
        # print(self.x.shape)
        # print(self.coef.shape)
        sum : float = 0
        result = (val_set_y - (val_set_x @ self.coef))**2
        for i in range(val_set_y.shape[0]):
            sum += result[i]
        return sum / val_set_y.shape[0]
    
    def least_squares_gradient(self) -> np.ndarray:
        # 2X^T * (Xw - Y)
        return ((2 * self.x.T) @ ((self.x @ self.coef) - self.y)) + (2 * self.L2 * self.coef) + (self.L1 * np.sign(self.coef))

    def predict(self, input : np.ndarray) -> np.ndarray:
        return self.coef @ input

    def print_coef(self, weight_names : list[str]) -> None:
        print(f'{'BIAS':14}', '(w0): ', self.coef[0])
        for i in range(self.x.shape[1] - 1):
            print(f'{weight_names[i]:14} (w{i+1}): ', self.coef[i+1])

# casas_dev : pd.DataFrame = prepro.correct_data_types(data_handler.RawData().casas_dev)
# casas_dev = prepro.convert_area_units(casas_dev, 'm2')
# casas_dev = prepro.remove_na_rows(casas_dev)
# casas_dev = prepro.standarize_numeric_columns(casas_dev)

# # lin : LinealReg = LinealReg(casas_dev[casas_dev['lat'] < 0]['area'].to_numpy(), casas_dev[casas_dev['lat'] < 0]['price'].to_numpy(), L1=10)
# # lin : LinealReg = LinealReg(casas_dev['area'].to_numpy(), casas_dev['price'].to_numpy(), L1=10)
# # lin.fit_pseudo_inverse()
# # lin.fit_gradient_descent(step_size=0.00005, tolerance=10000, max_number_of_steps=-1)
# # lin.fit_gradient_descent(step_size=0.0005, tolerance=0.5, max_number_of_steps=-1)
# # lin.print_coef()
# lin : LinealReg = LinealReg(casas_dev[['area', 'lat']].to_numpy(), casas_dev['price'].to_numpy())
# lin.fit_gradient_descent(step_size=0.00005, tolerance=0.5, max_number_of_steps=-1)
# lin.print_coef(weight_names=['area', 'lat'])

# plt.scatter(casas_dev['area'], casas_dev['price'], edgecolors='lightcyan')
# plt.scatter(casas_dev[casas_dev['lat'] < 0]['area'], casas_dev[casas_dev['lat'] < 0]['price'], edgecolors='lightcyan')
# plt.xlabel('area')
# plt.ylabel('price')

# a = np.arange(-2, 5, step=0.1)
# b = [lin.coef[0] + (lin.coef[1] * x) for x in a]
# plt.plot(a, b)

# plt.show()

# # a = np.array([[1, 3], 
# #               [3, 4], 
# #               [1, 4]])
# # b = np.array([1, 2])


# # print(a @ b)
# # print(a.shape)
# # print(b.shape)