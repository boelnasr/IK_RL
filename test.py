#!/usr/bin/env python3

import matplotlib
import matplotlib.pyplot as plt

# Try using the TkAgg or Agg backend depending on your environment.
# TkAgg is more suitable for interactive sessions, while Agg is good for saving images without displaying.
matplotlib.use('Qt5Agg')  # Use 'TkAgg' for interactive plots or 'Agg' for non-interactive
# Example plot
plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Sample Plot")

# Display the plot
plt.show()

