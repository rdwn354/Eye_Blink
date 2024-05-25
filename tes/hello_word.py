import matplotlib.pyplot as plt

# Create a new figure
plt.figure()

# Use the text method to display "Hello World"
plt.text(0.5, 0.5, 'Hello World', fontsize=20, ha='center', va='center')

# Adjust the axis to make sure the text is centered and visible
plt.xlim(0, 1)
plt.ylim(0, 1)

# Remove the axes for better visibility
plt.axis('off')

# Display the plot
plt.show()
