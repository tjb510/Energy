import numpy as np
import matplotlib.pyplot as plt

# Generate some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y)

# Add labels and title
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Sine wave")

# Show the plot
plt.show()