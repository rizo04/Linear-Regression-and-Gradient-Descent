# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 01:53:18 2018

@author: Flooki
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def gradient_over_each_example_set(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = gradient_over_each_example_set(b, m, points, learning_rate)
    return [b, m]

def main():
    points = np.genfromtxt("dataset.csv", delimiter=',')
    
    X = points[::,0]
    Y = points[::,1]
   
    learning_rate = 0.00000001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 100
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    
    x1 = 0
    y1 = m*x1+b
    x2 = 4700
    y2 = m*x2+b
    
    plt.figure(figsize = (15,4),dpi=100)
    plt.subplot(121)
    plt.scatter(X,Y,s=60,c='red',marker='*')
    plt.title('Relationship between house size and their prices')
    plt.xlabel("X - house size in square feet")
    plt.ylabel("Y - prices")
    plt.plot([x1, x2], [y1, y2], '-')
    plt.show()
    
    plt.hist(X,Y)
    plt.xlabel('size')
    plt.ylabel('price')
    plt.title('histogram of size and price')
    plt.axis([0,4000,0,700000])
    plt.grid(True)
    plt.show()

    
    
if __name__== "__main__":
  main()