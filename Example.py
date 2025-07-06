# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:23:31 2024
@author: krane
"""
from  numpy import *
import  matplotlib.pyplot as plt
from PoissonSolverwT import PoissonSolver
import sympy as sym

x, y = sym.symbols('x,y')

"--- To test the solve, define the solution here. Otherwise, modify the right-hand side function directly---"
 
u = sym.sin(x*y) #This is the true solution, use sympy to differentiate
u_func = sym.lambdify((x, y), u)


a = 1 #The constant alpha for the diffusion equation

f = sym.simplify(-a*(sym.diff(u, x, x)+sym.diff(u, y, y))+u) #Comment this line if no true solution can be given
f_func = sym.lambdify((x, y), f) #Define a function here for the right-hand side of the diffusion equation

I = u_func #The boundary condition
x0 = 0; xf = 1 # x domain
y0 = 0; yf = 1 # y domain

TestRefined = PoissonSolver(a, f_func, 10, 10, I=I,  Lx=x0, Rx=xf, Ly=y0, Ry=yf)
TestRefined.solver(central_difference=False)
TestRefined.grid_plot()

#TestRefined.true_plot(u_func) # For plotting of true solution, if known
#err=TestRefined.get_trueError(u_func)

# TestRefined.details(f) #For explicit details of the cells
# TestRefined.f_plot() #For plotting the right-hand side function on the domain
# TestRefined.error_plot() #For plotting the error, only works if true solution is known

   