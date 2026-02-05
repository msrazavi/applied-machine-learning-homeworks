import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define Function and Derivative
def f(x, y):
    """
    The objective function: f(x,y) = (x-3)^2 + (y+2)^2
    """
    return (x - 3)**2 + (y + 2)**2

def grad_f(x, y):
    """
    Gradient of f(x,y).
    Partial wrt x: 2(x-3)
    Partial wrt y: 2(y+2)
    """
    df_dx = 2 * (x - 3)
    df_dy = 2 * (y + 2)
    return np.array([df_dx, df_dy])

# Step 2: Implement Gradient Descent
def gradient_descent(start_x, start_y, learning_rate, n_iterations=20):
    """
    Performs gradient descent optimization.
    Returns a history of points (path).
    """
    # Initialize point
    x, y = start_x, start_y
    path = [(x, y)]
    
    for _ in range(n_iterations):
        # Calculate gradient
        grad = grad_f(x, y)
        
        # Update rule: x_new = x_old - alpha * gradient
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        
        # Store new point
        path.append((x, y))
        
    return np.array(path)

# Step 3: Visualize Optimization Path
def plot_results():
    # Simulation Parameters
    start_points = [(0, 0), (5, -5)]
    learning_rates = [0.05, 0.1, 0.5]
    iterations = 20
    true_min = (3, -2)

    # Setup the plot grid (2 rows, 3 columns)
    fig, axes = plt.subplots(len(start_points), len(learning_rates), figsize=(18, 10))
    
    # Adjust layout to make room for headers
    plt.subplots_adjust(left=0.1, top=0.9, bottom=0.1, hspace=0.4, wspace=0.3)

    # Prepare meshgrid for contour background
    x_range = np.linspace(-1, 6, 100)
    y_range = np.linspace(-6, 2, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)

    # Loop through Initial Points (Rows)
    for i, start_pt in enumerate(start_points):
        # Loop through Learning Rates (Columns)
        for j, lr in enumerate(learning_rates):
            ax = axes[i, j]
            
            # 1. Run Gradient Descent
            path = gradient_descent(start_pt[0], start_pt[1], lr, iterations)
            
            # 2. Draw Contour Plot
            ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
            
            # 3. Plot the Gradient Descent Path
            # Plot the path line and dots
            ax.plot(path[:, 0], path[:, 1], 'o-', color='red', label=f'lr={lr}', markersize=4, linewidth=1)
            
            # 4. Mark the True Minimum
            ax.scatter(true_min[0], true_min[1], color='green', marker='*', s=150, zorder=5, label='Minimum')
            
            # Styling the specific subplot
            ax.set_title("Gradient Descent Path", fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_xlim([-1, 6])
            ax.set_ylim([-6, 2])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, linestyle='--', alpha=0.3)

            # Add Column Titles (only on top row)
            if i == 0:
                ax.text(0.5, 1.15, f"Learning Rate = {lr}", transform=ax.transAxes, 
                        ha='center', fontsize=14, fontweight='bold')

    # Add Row Labels (Initial Points)
    rows = ["Initial Point\n(0, 0)", "Initial Point\n(5, -5)"]
    for ax, row_label in zip(axes[:,0], rows):
        ax.annotate(row_label, xy=(0, 0.5), xytext=(-0.4, 0.5),
                    xycoords='axes fraction', textcoords='axes fraction',
                    fontsize=14, ha='center', va='center', fontweight='bold', rotation=0)

    plt.suptitle(f"Homework 4: Minimizing f(x,y) = (x-3)^2 + (y+2)^2", fontsize=16, y=0.98)
    plt.show()

if __name__ == "__main__":
    plot_results()
