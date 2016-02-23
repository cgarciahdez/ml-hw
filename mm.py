import numpy as np
from pylab import *
from sklearn.cross_validation import KFold
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
import time

#
#This function loads a file with the name given into to np arrays x and y.
#Parameters: Name of the file to load
#Returns: nparray x with feature information and nparray y with label information
def load_file(name):
    file = open(name,'r')
    x=[]
    y=[]
    for line in file:
        if not line.startswith('#'):
            l=line.split()
            xi=[]
            for i in range(0,len(l)-1):
                xi.append(float(l[i]))
            x.append(xi)
            y.append(float(l[len(l)-1]))
    return np.array(x), np.array(y)

#This function calculates y-hat given x values, theta values and a degree.
#Parameters: x - matrix or vector, t - theta vector, d - degree
#Returns: y-hat vector
def yhat(x,t,d=1,gauss=False):
    if gauss:
        return dot(x,t.T)
    poly = PolynomialFeatures(degree=d)
    if(x.shape[1]>x.shape[0]):
        x=np.transpose(x)
    x=poly.fit_transform(x)
    return dot(x,t)

#This function regresses the data to find the polynomial model given a degree
#Parameters: x - vector or matrix, y - vector, d - degree of polynomial, two by default
#Returns: theta vector
def poly_reg(x,y,d=2):
    poly = PolynomialFeatures(degree=d)
    z = poly.fit_transform(x)
    a = dot(dot(inv(dot(z.T,z)),z.T),y)
    return a

#This is the iterative method to find 
def newton_method(x,y,d=2):
    poly = PolynomialFeatures(degree=d)
    Z=poly.fit_transform(x)
    theta=np.ones(Z.shape[1])
    theta=theta.reshape(len(theta),1)
    diff = 1
    while(diff > 1e-15):
    #for i in range (0,50):
        theta_i=theta
        theta = calc_theta(theta_i,Z,y)
        diff=np.absolute(theta_i-theta)
        diff=(np.average(diff))
    return theta
        
    
def calc_theta(t,Z,y):
    y=y.reshape(len(y),1)
    y_hat = np.dot(Z,t)
    ret = np.subtract(y_hat,y)
    ret = np.dot(np.transpose(Z),ret)
    h = np.dot(np.transpose(Z),Z)
    h = inv(h)
    ret = np.dot(h,ret)
    ret = t-ret
    return ret

#
#This function computes the MSE error given data and a theta vector
#Parameters: x - vector or matrix, y - vector, t - theta vector, d - degree of polynomial, one by default,
#gauss - determines wether the function to be analized is gaussian.
#Returns: MSE 
def compute_error(x,y,t,d=1,gauss=False):
    total =0
    _yhat=(yhat(x,t,d,gauss))
    for i in range(0,len(x)):
        total+=(((y[i]-_yhat[i])**2))
    total/=len(x)
    return total

#
#This function performs crossvalidation to find the average training and testing MSE's of
#given data using polynomial function, given a degree and a k-factor.
#Parameters: x - vector or matrix, y - vector, k - integer, d - degree, integer.
#Returns: training and testing MSE's
def cross_validation_multi(x,y,k=10,d=2):
    traine=0
    teste=0
    kf = KFold(x.shape[0],k)
    for train, test in kf:
        t = poly_reg(x[train],y[train],d)
        traine+=compute_error(x[train],y[train],t,d)
        teste += compute_error(x[test],y[test],t,d)
    traine/=k
    teste/=k
    
    return traine, teste

#
#This function performs crossvalidation to find the average training and testing MSE's of
#given data using the newton function, given a degree and a k-factor.
#Parameters: x - vector or matrix, y - vector, k - integer, d - degree, integer.
#Returns: training and testing MSE's
def cross_validation_newton(x,y,k=10,d=2):
    traine=0
    teste=0
    kf = KFold(x.shape[0],k)
    for train, test in kf:
        t = newton_method(x[train],y[train],d)
        traine+=compute_error(x[train],y[train],t,d)
        teste += compute_error(x[test],y[test],t,d)
    traine/=k
    teste/=k
    
    return traine, teste

#
#This function performs crossvalidation to find the average training and testing MSE's of
#given data using gaussian function, given a degree and a k-factor.
#Parameters: x - vector or matrix, y - vector, k - integer, d - degree, integer.
#Returns: training and testing MSE's
def cross_validation_gaussian(x,y,k=10):
    traine=0
    teste=0
    kf = KFold(x.shape[0],k)
    for train, test in kf:
        t = gaussian(x[train],y[train])
        t=t.T
        traine+=compute_error(x[train],y[train],t,gauss=True)
        teste +=compute_error(x[test],y[test],t,gauss=True)
    traine/=k
    teste/=k
    
    return traine, teste

#
#This function calculates the g matrix iterating through thetas
#Parameters: x - matrix
#Returns: g matrix
def calculate_g(x):
    row,col=x.shape
    g=np.ones((row,row))
    sigma=0.5
    for r in range(0,row):
        for c in range(0,row):
            g[r][c]=np.exp(-np.linalg.norm(x[r,:]-x[c,:])**2/2*sigma**2)
    return g

#
#This function calculates alpha as the dot product of g and y
#Parameters: g- matrix, y - vector
#Returns: alpha vector
def calculate_alpha(g,y):
    return dot(inv(g),y)

#This method calculates the theta coefficients with the gaussian kernel method.
#Parameters: x - matrix, y - vector
#Returns: theta vector
def gaussian(x,y):
    g = calculate_g(x)
    alpha = calculate_alpha(g,y)
    theta = np.dot(np.transpose(alpha),x)
    return theta
    
    
#
#This function runs the program according to the given parameters, and prints all necessary
#results required from the homework.
#Parameters: name - name of file being analized
def run(name,maxd):
    arr=load_file(name)
    x,y=arr


    for d in range(2,maxd+1):
        print("For degree %d:\n"%d)
        start = time.time()
        c1 = cross_validation_multi(x,y,10,d)
        end1 = time.time()
        
        start = time.time()
        c2 = cross_validation_newton(x,y,10,d)
        end2 = time.time()

        types = ["          Explicit          ","          Iterative          "]
        errors = ["Training error   ","Testing error    "]
        data = np.array([c1,c2])

        print ("Time for explicit method was: %f\nTime for iterative method was: %f\n"%(end1,end2))

        row_format ="{:>15}" * (len(types) + 1)
        print (row_format.format("", *types))
        for eror, row in zip(errors, data):
            print (row_format.format(eror, *row))
    
    print("Gaussian kernel:")
    start = time.time()
    c = cross_validation_gaussian(x,y)
    end = time.time()
    print ("Time elapsed was: %f"%(end-start))
    print ("Training error: %f\nTesting error: %f\n"%(c[0],c[1]))

r=load_file("mvar-set1.dat.txt")
#print (newton_method(r[0],r[1],2))
a = np.arange(20).reshape((5,4))
#print(calculate_g(r[0]))
#print(a[:,[0,1]])
#print ((r[0]))
#print ((r[1]))
#print(poly_reg(r[0],r[1],2))
#print(gaussian(r[0],r[1]))

run("mvar-set3.dat.txt",4)
