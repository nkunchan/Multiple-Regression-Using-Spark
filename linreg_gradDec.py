#nkunchan@uncc.edu
#Nikita Kunchanwar
#800962459



# linreg_gradDec.py
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

# gradDec function which implements gradient decent approach to calculate beta value using alpha and no of iterations from command line argument
def gradDec(yxlines,beta,alpha,no_of_iter):    
	itr=0
	while itr<no_of_iter:
		itr=itr+1;    #increment iteration
		# calculate the term to be added to beta calling function "updateTerm" for all entries in dataset using map function and sum them up in reduce 
  		updatedTerm_value=yxlines.map(lambda x_y:("updateTerm",updateTerm(x_y,beta,alpha))).reduceByKey(lambda term1,term2:term1+term2)
  		sum2=updatedTerm_value.map(lambda final_value:final_value[1]).collect()[0]  #action for evaluation
		# add the above term to beta to get new beta to be used in next iteration
		beta=beta+sum2

                # This code is to check whether beta values are getting converged and continue the loop until it converges by calculating difference of errors or reach the maximum iterations         
		#fraction=0.0001     #term to compare the difference of errors after beta value is updated
		#converged=False 

                #while not converged:
		
		#prev_error=yxlines.map(lambda x_y:("error",error(x_y,beta))).reduceByKey(lambda term1,term2:((term1))+((term2))).map(lambda final_value:final_value[1]).collect()[0]
		#print "preverror",prev_error
		
		#above code to update beta

		#current_error=yxlines.map(lambda x_y:("error",error(x_y,beta))).reduceByKey(lambda term1,term2:((term1))+((term2))).map(lambda final_value:final_value[1]).collect()[0]
		#print "currenterror",current_error
		#if abs(round(prev_error,2)-round(current_error,2))<=fraction:
			#converged=True
			#print "iter",itr
			
		#if itr>=1000:
			#converged=True
			#print "iter exceeded"
		
	return beta


#function to calculate the term to be added to beta 
def updateTerm(x_y,beta,alpha):
        temp_y=float(x_y[0])  #fetch y value from x and y values
        x_y[0]=1   #fix first term as 1 for x values
        array_x=np.array(x_y).astype('float')
        X=np.asmatrix(array_x)   #create matrix with x values
        mul_xBeta=np.dot(X,beta) #calculate x*beta
	sub_term=(temp_y-mul_xBeta) #calculate (y-x*beta)
        del_term=np.dot(X.T,sub_term)  #calculate (x transpose * (y-(x*beta)))
        return alpha*del_term  #return alpha * above term


# Function to calculate error in prediction (will be used for convergence logic) 
def error(x_y,beta):
	temp_y=float(x_y[0])
        x_y[0]=1
        array_x=np.array(x_y).astype('float')
        X=np.asmatrix(array_x)
        mul_xBeta=np.dot(X,beta)
	sub_term=(temp_y-mul_xBeta)
	return sub_term
        
  


if __name__ == "__main__":
  if len(sys.argv) !=4:
    print >> sys.stderr, "Usage: linreg <datafile> <alpha> <# of iterations>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Read Second argument from command line which is input file
  yxinputFile = sc.textFile(sys.argv[1])


  # Assign value given in the third argument through command line to alpha
  alpha=float(sys.argv[2]);

  # Assign value given in the fourth argument through command line to no_of_iterations
  no_of_iter=int(sys.argv[3]);
	

  yxlines = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = yxlines.first()
  yxlength = len(yxfirstline)


  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros(yxlength, dtype=float).reshape(yxlength,1)
  
  # call gradDec function to calculate beta using gradient descent approach
  beta=gradDec(yxlines,beta,alpha,no_of_iter)
 

  # print the linear regression coefficients in desired output format
  print "beta: "
  for coeff in beta:
      print coeff

  sc.stop()
