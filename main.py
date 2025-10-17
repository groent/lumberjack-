import numpy as np 
import matplotlib.pyplot as plt

# Task 1 
# In this task we approximate ln(2) by using the theory provided in the assigment. 
 

# We start by creating the function we were instructed to create, and initializing the variables were required to initiliaze.
def approx_ln(x,n):
    a_0 = (1+x)/2 
    g_0 = np.sqrt(x)

    # In this for loop we iterate, the approximation by the specified "n" amount of steps. 
    for i in range(1,n+1):
        # Since our starting value is a_0, we express "a_i+1" as a_i using the given formula. We then replace the starting value a_0 with a_i and then continue. The same applies for g_0 and g_i. 
        a_i = (a_0 + g_0) / 2 
        g_i = np.sqrt(a_i * g_0)
        a_0 = a_i
        g_0 = g_i 
    return (x-1)/a_0

print(f"Approx: {approx_ln(2,100)}, numpy: {np.log(2)} ")

# Task 2 

# Here we get out x values, we iterate by one to get 100 values 
x = np.linspace(1,101,101)

# To display both the plots we use the subplot function
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

ax1.set_title('Numpy ln(x) and approx_ln')

# We plot our y-values 
ax1.plot(x, np.log(x), label = "numpy_ln")
# We plot of a given n-value and itterate up to a certain range 
for i in range(1,6):
    ax1.plot(x, approx_ln(x, i), label = f"approx_ln_{i}")
ax1.legend()

ax2.set_title("Difference between numpy ln and approx_ln")
# We plot the error by plotting the difference between numpy's "ln" function to our "approx_ln" functions 
# We plot for a step size and then iterate up to a certain range.
for i in range(1,6):
    ax2.plot(x, np.log(x) - approx_ln(x, i), label = "approx_ln_1")
ax2.legend()

plt.show()

#Task 3 

# Make an empty list to store the y error values.
y_approx_error = []
# Iterate 101, error values to match the length of the x linspace
for i in range(1,len(x)+1):
    # append the absolute subtraction of the numpy approximation with our approximation.
    y_approx_error.append(float(np.abs(np.log(1.41) - approx_ln(1.41 ,i))))

# Display the plot seperately. 
plt.title("Absolute value of the Error vs. n for the case x= 1.41")
plt.plot(x, y_approx_error )
plt.show()

# Task 4
def fast_approx(x,n):
    # Initialize main variables
    a_0 = (1+x)/2 
    g_0 = np.sqrt(x)
    # d_x represents a step in the recursions process. This is the initial step.
    d_0i = (a_0 + g_0) /2 
    # This is the previous step which we store as a variable that will be used later on in the algorithm
    d_kmin1imin1 = 0
    # This is the next step in the algorithm 
    d_ki = 0
    # In this for loop we iterate, the approximation by the specified "n" amount of steps. 
    for i in range(0,n+1):
        # Since our starting value is a_0, we express "a_i+1" as a_i using the given formula. We then replace the starting value a_0 with a_i and then continue. The same applies for g_0 and g_i. 
        a_i = (a_0 + g_0) / 2 
        g_i = np.sqrt(a_i * g_0)
        a_0 = a_i
        g_0 = g_i 
        # Save the previous step of d_x for future use. 
        d_kmin1imin1 = d_0i
        # Define d_x as the current step of the algorithm 
        d_0i = a_i
        #Using the given formula and variables determine the next step
        for j in range(1, n+1):
            d_ki = (d_0i-(4**(-j))*d_kmin1imin1)/(1-(4**(-j)))
    return (x-1)/d_ki

print(f"Fast_approx = {fast_approx(2,10)}")

 
#  Task 5
# Make a separate x linspace
x_2 = np.linspace(0,20,100)

for i in range(0,6):
    y_fast_approx_error = []
    #Iterate as many times as the length of x_2, error values to match the length of the x linspace
    for j in range(0,len(x_2)):
        # append the absolute subtraction of the numpy approximation with our approximation.
        y_fast_approx_error.append(float(np.abs(np.log(x_2[j]) - fast_approx(x_2[j] ,i))))
    plt.scatter(x_2,y_fast_approx_error, label= f"iteration {i}")

plt.title("Error behaviour of the Accelerated Carlsson method fo the log")
plt.xlim([0, 20])
plt.yscale('log')
plt.legend()
plt.show()