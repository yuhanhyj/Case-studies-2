import numpy as np
import matplotlib.pyplot as plt

def read_residuals(filename):
    """
    Reads residual data from a file. The file is assumed to have the following format:
    
    n = 8
    0 0.12345
    1 0.05678
    2 0.00123
    ...
    
    (an empty line separates different system sizes)
    
    Returns a dictionary with keys as the system size (n) and values as lists of residuals.
    """
    data = {}
    with open(filename, 'r') as f:
        current_n = None
        current_list = []
        for line in f:
            line = line.strip()
            if not line:
                if current_n is not None:
                    data[current_n] = current_list
                    current_n = None
                    current_list = []
                continue
            if line.startswith("n ="):
                # New system size line, e.g., "n = 8"
                parts = line.split("=")
                current_n = int(parts[1].strip())
            else:
                # Expecting a line like "0 0.12345" or "iter 0: 0.12345"
                parts = line.split()
                # Try to parse the first part as iteration number and the second as the residual.
                try:
                    # If the line has a colon, remove it.
                    iter_num = int(parts[0].replace("iter", "").replace(":", ""))
                    res = float(parts[1])
                    current_list.append(res)
                except (ValueError, IndexError):
                    # Skip any improperly formatted lines.
                    continue
        # Catch any data at EOF
        if current_n is not None and current_list:
            data[current_n] = current_list
    return data

def main():
    # Filename containing residual data
    filename = "gmres_residuals.txt"
    data = read_residuals(filename)

    plt.figure(figsize=(10, 6))
    
    # Plot data for each system size
    for n, residuals in data.items():
        iterations = list(range(len(residuals)))
        plt.semilogy(iterations, residuals, marker='o', label=f"n = {n}")
    
    # Add first and second order convergence reference lines
    if len(data) > 0:
        first_residuals = next(iter(data.values()))
        r0 = first_residuals[0]
        x = np.array([0, 128])
        
        # Second order convergence: O(2^(-x²))
        y2 = r0 * 4.0**(-x)
        plt.semilogy(x, y2, 'k:', label='Second Order O(4^(-n))')
        print(y2)
    
    plt.xlabel("Iteration Number n")
    plt.ylabel("Normalized Residual ||rₖ|| / ||b||")
    plt.title("GMRES Convergence: Normalized Residual vs. Iteration")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.ylim(1e-18, 1.0)  # Set y-axis range from 10⁻¹⁶ to 1
    plt.tight_layout()
    
    plt.savefig('gmres_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
