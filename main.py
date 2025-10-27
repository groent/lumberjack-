import numpy as np 
import matplotlib.pyplot as plt


# Task 1 
# In this task we approximate ln(2) by using the theory provided in the assigment. 
 

# We start by creating the function we were instructed to create, and initializing the variables were required to initiliaze.
def approx_ln(x,n):
    """ 
    This function approximates the natural log of a give "x", an "n" number of iterations. 
    """
    a = (1+x)/2 
    g = np.sqrt(x)

    # In this for loop we iterate, the approximation by the specified "n" amount of steps. 
    for i in range(1,n+1):
        a = (a + g) / 2 
        g = np.sqrt(a * g)
    return (x-1)/a

print(f"Approx: {approx_ln(2,100)}, numpy: {np.log(2)} ")


# Task 2 

# Here we get out x values, we iterate by one to get 100 values 
x = np.linspace(1,101,101)

# To display both the plots we use the subplot function
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

ax1.set_title("Numpy ln(x) and approx_ln")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
# Plot y values of np.log and approx_ln for each point in x
ax1.plot(x, np.log(x), label = "numpy_ln")
for i in range(1,6):
    ax1.plot(x, approx_ln(x, i), label = f"approx_ln_{i}")
ax1.legend()

ax2.set_title("Difference between numpy ln and approx_ln")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
# Plot the error between np.log and approx_ln
for i in range(1,6):
    ax2.plot(x, np.log(x) - approx_ln(x, i), label = f"approx_ln_{i}")
ax2.legend()
ax2.set_yscale("log")
plt.tight_layout()
plt.show()


#Task 3 

# Make an empty list to store the y error values.
y_approx_error = []
# Calculate and store the error for each x value in x
for i in range(1,len(x)+1):
    y_approx_error.append(float(np.abs(np.log(1.41) - approx_ln(1.41 ,i))))

# Display the plot seperately. 
plt.title("Absolute value of the Error vs. n for the case x= 1.41")
plt.plot(x, y_approx_error )
plt.xlabel("x")
plt.ylabel("y")
plt.yscale("log")
plt.show()


# Task 4
def fast_approx(x,n):
    """ 
    Uses the given recurssion formula to approximate the natural logarithm of "x", for "n" iterations.  
    """
    a = (1+x)/2 
    g = np.sqrt(x)
    d = [[a]]
    # Iterate a and store the values for the list d to get the row d_0i
    for i in range(0,n+1):
        a = (a + g) / 2 
        g = np.sqrt(a * g) 
        d[0].append(a)
    
    # Calculate and store each value in a nested list for each d_ki
    for j in range(0,i+2):
        d.append([])
        for k in range(1, len(d[j])):
            d[j+1].append((d[j][k]-4**(-(j+1))*d[j][k-1])/(1-4**(-(j+1))))
    return (x-1)/d[n-1][0]

 
#  Task 5
# Obtain x values 
x_2 = np.linspace(0,20,100)

for i in range(1,7):
    y_fast_approx_error = []
    # Calculate the error and store the value for each x value in x_2
    for j in range(0,len(x_2)):
        y_fast_approx_error.append(float(np.abs(np.log(x_2[j]) - fast_approx(x_2[j] ,i))))
    plt.scatter(x_2,y_fast_approx_error, label= f"iteration {i}")

plt.title("Error behaviour of the Accelerated Carlsson method fo the log")
plt.xlim([0, 20])
plt.yscale("log")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper left")
plt.show()