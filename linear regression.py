import numpy as np

def linear(x,y,a):
    sum_x = np.sum(x)
    # print(sum_x)
    sum_x2 = np.sum(x ** 2)
    # print(sum_x2)
    sum_y = np.sum(y)
    # print(sum_y)
    sum_xy = np.sum(x * y)
    # print(sum_xy)
    n = len(x)
    beta0 = ((sum_y * sum_x2) - (sum_x * sum_xy)) / (n * sum_x2 - sum_x ** 2)
    beta1 = ((n * sum_xy - (sum_x * sum_y)) / (n * sum_x2 - sum_x ** 2))
    print(f"Equation of the data is: y = {beta1:.2f}x + {beta0:.2f}")

    req_y = beta0 + beta1 * a

    return req_y
x = input("Enter numbers separated by spaces: ")
x_array = np.array([float(i) for i in x.split()])
y = input("Enter numbers separated by spaces: ")
y_array=np.array([float(i) for i in y.split()])
z=int(input("Enter required z value: "))
result=linear(x_array,y_array,z)
print(f"The predicted value of y for x = 55 is: {result:.2f}")