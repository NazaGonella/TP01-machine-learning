import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import preprocessing as prepro
from utils import RawData

class LinealReg:
    def __init__(self, x : np.ndarray, y : np.ndarray, initial_weight_value : float = 1):
        self.x : np.ndarray = np.c_[np.ones(x.shape[0]), x]     # agrego columna de unos para el bias.
        self.y : np.ndarray = y
        self.coef : np.ndarray = np.full(shape=self.x.shape[1], fill_value=initial_weight_value)
    
    def fit_pseudo_inverse(self):
        # w = (X^T * X)^-1 * X^T * Y
        self.coef = np.matmul(np.matmul(np.linalg.pinv(np.matmul(self.x.T, self.x)), self.x.T), self.y)

    def fit_gradient_descent(self, step_size : float, tolerance : float, max_number_of_steps : int = -1):
        attempts = 0
        while True:
            gradient = self.least_squares_gradient()
            if np.linalg.norm(gradient) <= tolerance or (attempts >= max_number_of_steps and max_number_of_steps != -1):
                break
            # print("(step_size * (gradient / np.linalg.norm(gradient)): ", (step_size * (gradient / np.linalg.norm(gradient))))
            # print("(gradient / np.linalg.norm(gradient): ", (gradient / np.linalg.norm(gradient), "\n"))
            print(self.error_cuadratico_medio())
            # self.coef = self.coef - (step_size * (gradient / np.linalg.norm(gradient)))
            self.coef = self.coef - (step_size * (gradient))
            attempts += 1

    def error_least_squares_function(self) -> float:
        # ||Xw - Y||^2
        return np.linalg.norm((self.x @ self.coef) - self.y)**2

    def error_cuadratico_medio(self) -> float:
        sum : float = 0
        result = (self.y - (self.x @ self.coef))**2
        for i in range(self.y.shape[0]):
            sum += result[i]
        return sum / self.y.shape[0]
    
    def least_squares_gradient(self) -> np.ndarray:
        # 2X^T * (Xw - Y)
        return (2 * self.x.T) @ ((self.x @ self.coef) - self.y)

casas_dev : pd.DataFrame = prepro.correct_data_types(RawData.casas_dev)
casas_dev = prepro.convert_area_units(casas_dev, 'm2')

lin : LinealReg = LinealReg(casas_dev[casas_dev['lat'] > 0]['price'].to_numpy(), casas_dev[casas_dev['lat'] > 0]['area'].to_numpy())
# lin.fit_pseudo_inverse()
# lin.fit_gradient_descent(step_size=0.00005, tolerance=10000, max_number_of_steps=-1)
lin.fit_gradient_descent(step_size=0.0000000005, tolerance=5000, max_number_of_steps=-1)

plt.scatter(casas_dev['price'], casas_dev['area'], edgecolors='lightcyan')
plt.scatter(casas_dev[casas_dev['lat'] > 0]['price'], casas_dev[casas_dev['lat'] > 0]['area'], edgecolors='lightcyan')
plt.xlabel('price')
plt.ylabel('area')

a = np.arange(2000, step=10)
b = [lin.coef[0] + (lin.coef[1] * x) for x in a]
plt.plot(a, b)

plt.show()

# a = np.array([[1, 3], 
#               [3, 4], 
#               [1, 4]])
# b = np.array([1, 2])


# print(a @ b)
# print(a.shape)
# print(b.shape)