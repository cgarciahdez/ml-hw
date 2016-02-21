
from matplotlib import pyplot as plt
import numpy as np
from pylab import *
from sklearn.cross_validation import KFold
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures


def load_file(name):
    file = open(name,'r')
    x=[]
    y=[]
    for line in file:
        if not line.startswith('#'):
            l=line.split()
            x.append(float(l[0]))
            y.append(float(l[1]))
    return np.array(x), np.array(y)


def plot(x,y,t=None): 
    plt.plot(x,y,'go')
    
    if t is not None:
        t=t.reshape(len(t),1)
        start=int(min(x))
        stop=int(max(x))
        Z=np.ones(stop-start)
        X=np.arange(start,stop)
        for j in range(1,len(t)):
            Z=np.column_stack((Z,X))
            X=X*X
        y_hat=np.dot(Z,t)
        plt.plot([float(i) for i in list(range(int(start),int(stop)))],y_hat,'r')
                
    plt.show()

def yhat(x,t,d=1):
    poly = PolynomialFeatures(degree=d)
    if len(x.shape)>1:
        x=np.transpose(x)
    else:
        x=x.reshape(len(x),1)
    x=poly.fit_transform(x)
    t = t.reshape(len(t),1)
    return dot(x,t)
                
                
def simple_linreg(x,y):
    m = len(x)
    sum_x = x.sum()
    sum_x2= (x**2).sum()
    A = np.array([[m,sum_x],[sum_x,sum_x2]])
    sum_y=y.sum()
    sum_xy=np.dot(x,y)
    b = np.array([sum_y,sum_xy])
    t=solve(A,b)
    return t

def py_simple_linreg(x,y):
    regr = linear_model.LinearRegression()
    x = x.reshape(len(x), 1)
    regr.fit(x,y)
    return regr

def compute_error(x,y,t,d=1):
    total =0
    _yhat=np.transpose(yhat(x,t,d))[0]
    for i in range(0,len(x)):
        total+=((y[i]-_yhat[i])**2)/(y[i]**2)
    total/=len(x)
    return total

def py_compute_error(yg,y):
    return np.mean(((y-yg) ** 2)/y**2)


def py_cross_validation(px,py,k,d=1):
    traine=0
    teste=0
    px=px.reshape(len(px),1)
    kf = KFold(len(px),k)
    for train, test in kf:
        regr = py_simple_linreg(px[train],py[train])
        traine+=py_compute_error(regr.predict(px[train]),py[train])
        teste += py_compute_error(regr.predict(px[test]),py[test])
    traine/=k
    teste/=k
    
    return traine, teste


def cross_validation(x,y,k=10,d=1):
    traine=0
    teste=0
    kf = KFold(len(x),k)
    for train, test in kf:
        t = poly_linreg(x[train],y[train],d)
        traine+=compute_error(x[train],y[train],t,d)
        teste += compute_error(x[test],y[test],t,d)
    traine/=k
    teste/=k
    
    return traine, teste
                

def poly_linreg(x,y,d=1):
    poly = PolynomialFeatures(degree=d)
    if len(x.shape)>1:
        x=np.transpose(x)
    else:
        x=x.reshape(len(x),1)
    Z = poly.fit_transform(x)
    t = dot(dot(inv(dot(Z.T,Z)),Z.T),y)
    return t

def compare(x,y,maxd,k=10):
    ret=[]
    for i in range(1,maxd+1):
        t = poly_linreg(x,y,i)
        traine,teste=cross_validation(x,y,k,i)
        ret.append((t,traine,teste,i))
    return sorted(ret, key=lambda x: (x[1]+x[2])/2)
    
    
def run(name,maxd,plotall=False):
    arr=load_file(name)
    x,y=arr
    print("Data plot:")
    plot(x,y)
    print("Linear regression:")
    print ("The RSS training error is: %f\nThe RSS testing error is: %f"%cross_validation(arr[0],arr[1],10))
    theta_=simple_linreg(x,y)
    print("The theta coefficients are: ")
    for i in range (0,len(theta_)):
        print ("theta_"+str(i)+": "+str(theta_[i]))
    print("Polynomial regressions comparison until degree %d:"%maxd)
    cp = compare(x,y,maxd)
    for c in cp:
        print ("Using degree %d.\nTraining error: %f\nTesting error: %f\n"%(c[3],c[1],c[2]))
    print ("The regression with degree %d has the smallest sum of errors. Plot: "%cp[0][3])
    plot(x,y,cp[0][0])
    if plotall:
        for i in range(1,len(cp)):
            print("Plot degree %d: "%cp[i][3])
            plot(x,y,cp[i][0])
    

    
    
arr=load_file("svar-set2.dat.txt")
#plot(arr[0],arr[1])
traine,teste=cross_validation(arr[0],arr[1],10)
#print((traine,teste))
print (py_cross_validation(arr[0],arr[1],10))
t = simple_linreg(arr[0],arr[1])
tp = py_simple_linreg(arr[0],arr[1])
rest= (poly_linreg(arr[0],arr[1],1))
te,tes=cross_validation(arr[0],arr[1],10,2)
#print (t)
#print(rest)
#print(str(tp.intercept_)+","+str(tp.coef_[0]))
#plot(arr[0],arr[1],t)
res= (poly_linreg(arr[0],arr[1],2))
#print(res)
#plot(arr[0],arr[1],res)
#print(te,tes)

run("svar-set1.dat.txt",4,True)

