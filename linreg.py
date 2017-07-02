#nkunchan@uncc.edu
#Nikita Kunchanwar
#800962459

# linreg.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
import numpy as np
from numpy.linalg import inv

from pyspark import SparkContext



# Function to calculate (X transpose * X)
def keyA(XT_X):
        XT_X[0]=1 #fix 1st term as 1 in x values
        array_x=np.array(XT_X).astype('float') #create array of float for x
        X=np.asmatrix(array_x).T #create matrix of x
        mul_xTx=np.dot(X,X.T) #calculate x * x transpose
        return mul_xTx  #return above calculated value

# Function to calculate (X * Y)
def keyB(x_y):
        y_value=float(x_y[0]) #fetch y value from x and y values
        x_y[0]=1 #fist 1st term as 1 in x values
        array_x=np.array(x_y).astype('float')  #create array of float for x
        X=np.asmatrix(array_x).T #create matrix of x
        mul_xy=np.dot(X,y_value) #calculate x *  Y
        
        return mul_xy #return above calculated value
        
 


if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)


  #Call function KeyA to calculate summation of X*XT using map reduce using all entries from input file
  #Map calculate individual x*xt for each entry and emit along with KeyA  
  #Reduce calculate summation of term from the valus got from map
  
  KeyA_value=yxlines.map(lambda X_XT:("KeyA",keyA(X_XT))).reduceByKey(lambda a1,a2:a1+a2)
  sum1= KeyA_value.map(lambda final_value:final_value[1]).collect()[0]   #action for evaluation
  A=np.matrix(sum1)
  A_inverse= inv(A) #calculate inverse of A
  
  #Call function KeyB to calculate summation of X*Y using map reduce using all entries from input file
  #Map calculate individual x*y for each entry and emit along with KeyB  
  #Reduce calculate summation of term from the valus got from map

  KeyB_value=yxlines.map(lambda x_y:("KeyB",keyB(x_y))).reduceByKey(lambda a1,a2:a1+a2)
  sum2=KeyB_value.map(lambda final_value:final_value[1]).collect()[0]   #action for evaluation
  B=np.matrix(sum2)
  
  #calculate A*B
  beta=np.dot(A_inverse,B)
 
   

  # dummy floating point array for beta to illustrate desired output format
  #beta = np.zeros(yxlength, dtype=float)

  #
  # Add your code here to compute the array of 
  # linear regression coefficients beta.
  # You may also modify the above code.
  #

  # print the linear regression coefficients in desired output format
  print "beta: "
  for coeff in beta:
      print coeff

  sc.stop()
