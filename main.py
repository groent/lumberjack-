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

# We get our y-values by applying our functions to the x-value for each itteration of our functions 
y_numpy_ln = (np.log(x))
y_approx_ln_1 = approx_ln(x,1)
y_approx_ln_2 = approx_ln(x,2)
y_approx_ln_3 = approx_ln(x,3)
y_approx_ln_5 = approx_ln(x,5)

# To display both the plots we use the subplot function
# f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)

# ax1.set_title('Numpy ln(x) vs approx_ln')

# We plot our y-values 
# ax1.plot(x, y_numpy_ln, label = "numpy_ln")
# ax1.plot(x, y_approx_ln_1, label = "approx_ln_1")
# ax1.plot(x, y_approx_ln_2, label = "approx_ln_2")
# ax1.plot(x, y_approx_ln_3, label = "approx_ln_3")
# ax1.plot(x, y_approx_ln_5, label = "approx_ln_5")
# ax1.legend()

# We plot the error by plotting the difference between numpy's "ln" function to our "approx_ln" functions 
# ax2.plot(x, y_numpy_ln - y_approx_ln_1, label = "approx_ln_1")
# ax2.plot(x, y_numpy_ln - y_approx_ln_2, label = "approx_ln_2")
# ax2.plot(x, y_numpy_ln - y_approx_ln_3, label = "approx_ln_3")
# ax2.plot(x, y_numpy_ln - y_approx_ln_5, label = "approx_ln_5")
# ax2.legend()

# plt.show()

#Task 3 

# Make an empty list to store the y error values.
y_approx_error = []
# Iterate 101, error values to match the length of the x linspace
for i in range(1,102):
    # append the absolute subtraction of the numpy approximation with our approximation.
    y_approx_error.append(float(np.abs(np.log(1.41) - approx_ln(1.41 ,i))))

# Display the plot seperately. 
# plt.plot(x, y_approx_error )
# plt.show()

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

# Task 5
# Make a separate x linspace
x_2 = np.linspace(1,20,100)

# Make an empty list to store the y error values.
y_acc_approx1_error = []
y_acc_approx2_error = []
y_acc_approx3_error = []
y_acc_approx4_error = []
y_acc_approx5_error = []
# Iterate 101, error values to match the length of the x linspace
for i in range(1,101):
    # append the absolute subtraction of the numpy approximation with our approximation.
    y_acc_approx1_error.append(float(np.abs(np.log(i) - fast_approx(i ,1))))
    y_acc_approx2_error.append(float(np.abs(np.log(i) - fast_approx(i ,2))))
    y_acc_approx3_error.append(float(np.abs(np.log(i) - fast_approx(i ,3))))
    y_acc_approx4_error.append(float(np.abs(np.log(i) - fast_approx(i ,4))))
    y_acc_approx5_error.append(float(np.abs(np.log(i) - fast_approx(i ,5))))
plt.scatter(x_2, y_acc_approx1_error, label = "Iter = 1" )
plt.scatter(x_2, y_acc_approx2_error, label = "Iter = 2" )
plt.scatter(x_2, y_acc_approx3_error, label = "Iter = 3" )
plt.scatter(x_2, y_acc_approx4_error, label = "Iter = 4" )
plt.scatter(x_2, y_acc_approx5_error, label = "Iter = 5" )
plt.yscale('log')
plt.legend()
plt.show()