/**
 * @file Arnoldi.cpp
 * @brief  Arnoldi iteration for GMRES
 *
 * This program implements the Arnoldi iteration, to generate an orthonormal basis for the Krylov subspace and the corresponding Hessenberg matrix.
 * @author Yujin Han
 * @date 2025-03-25
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

// Function to print a matrix
void printMatrix(const std::vector<std::vector<double>> &M)
{
    for (const auto &row : M)
    {
        for (double val : row)
        {
            std::cout << std::setw(12) << std::setprecision(6) << val << " ";
        }
        std::cout << std::endl;
    }
}

// Function to print a vector
void printVector(const std::vector<double> &V)
{
    for (double val : V)
    {
        std::cout << std::setw(12) << std::setprecision(6) << val << " ";
    }
    std::cout << std::endl;
}

// Function to check the orthogonality of Q:
// It prints the norm of each column (should be ~1)
// and the dot product between any two different columns (should be close to 0)
void checkOrthogonality(const std::vector<std::vector<double>> &Q)
{
    int n = Q.size();          // Dimension of the vectors
    int numCols = Q[0].size(); // Number of basis vectors (columns)
    std::cout << "\n--- Orthogonality Check ---" << std::endl;
    std::vector<std::vector<double>> QTQ(numCols, std::vector<double>(numCols, 0.0));
    for (int i = 0; i < numCols; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            for (int k = 0; k < n; k++)
            {
                QTQ[i][j] += Q[k][i] * Q[k][j];
            }
        }
    }

    std::cout << "The Q^T * Q matrix is: " << std::endl;
    printMatrix(QTQ);
}

// Function to check the Arnoldi relation: A * Q_m ≈ Q_{m+1} * H_m
// where Q_m consists of the first m columns of Q and Q_{m+1} is all columns of Q.
void checkArnoldiRelation(const std::vector<std::vector<double>> &A,
                          const std::vector<std::vector<double>> &Q,
                          const std::vector<std::vector<double>> &H,
                          int m)
{
    int n = A.size(); // A is n x n, Q is n x (m+1), H is (m+1) x m
    // Compute M = A * Q_m, where Q_m consists of the first m columns of Q
    std::vector<std::vector<double>> M(n, std::vector<double>(m, 0.0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int p = 0; p < n; p++)
            {
                M[i][j] += A[i][p] * Q[p][j];
            }
        }
    }
    // Compute MH = Q_{m+1} * H, where Q_{m+1} is all columns of Q
    std::vector<std::vector<double>> MH(n, std::vector<double>(m, 0.0));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int p = 0; p < m + 1; p++)
            {
                MH[i][j] += Q[i][p] * H[p][j];
            }
        }
    }
    // Compute the Frobenius norm ||M - MH||
    double error = 0.0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            double diff = M[i][j] - MH[i][j];
            error += diff * diff;
        }
    }
    error = std::sqrt(error);
    std::cout << "\n||A * Q_m - Q_{m+1} * H_m||_F = " << error << std::endl;
}

double norm2(const std::vector<double> &v)
{
    double sum = 0.0;
    for (double val : v)
        sum += val * val;
    return std::sqrt(sum);
}

/**
 * @brief Performs Arnoldi iteration according to Algorithm 1.1.
 * @param b Initial vector (length n)
 * @param A n x n matrix
 * @param k Number of iterations (produces k+1 orthogonal vectors; H is (k+1) x k)
 * @param tol Tolerance for the stopping condition
 * @return A pair containing H and Q.
 */
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> arnoldi(
    const std::vector<double> &b,
    const std::vector<std::vector<double>> &A,
    int k,
    double tol = 1e-8)
{
    int n = b.size();
    std::cout << "Matrix size: " << n << "x" << n << ", Iterations: " << k << std::endl;

    // Initialize Q (n x (k+1)) and H ((k+1) x k)
    std::vector<std::vector<double>> Q(n, std::vector<double>(k + 1, 0.0));
    std::vector<std::vector<double>> H(k + 1, std::vector<double>(k, 0.0));

    // Set Q[:,0] = b
    for (int i = 0; i < n; i++)
    {
        Q[i][0] = b[i];
    }

    // Normalize Q[:,0]
    double norm_b = norm2(b);

    if (norm_b == 0)
    {
        std::cerr << "Error: Initial vector b has zero norm!" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n; i++)
        Q[i][0] /= norm_b;

    // Begin Arnoldi iteration
    for (int j = 0; j < k; j++)
    {
        std::cout << "Iteration: " << j << std::endl;
        std::vector<double> v(n, 0.0);

        // Compute v = A * Q[:,j]
        for (int i = 0; i < n; i++)
        {
            for (int p = 0; p < n; p++)
            {
                v[i] += A[i][p] * Q[p][j];
            }
        }

        // Modified Gram-Schmidt orthogonalization
        for (int i = 0; i <= j; i++)
        {
            H[i][j] = 0.0;
            for (int p = 0; p < n; p++)
            {
                H[i][j] += Q[p][i] * v[p];
            }
            for (int p = 0; p < n; p++)
            {
                v[p] -= H[i][j] * Q[p][i];
            }
        }

        // Compute H[j+1][j] = ||v||
        const double h_next = norm2(v);

        H[j + 1][j] = h_next;

        // If h_next is too small, stop the iteration
        if (H[j + 1][j] < tol)
        {
            std::cerr << "Warning: Small H[j+1][j] = " << H[j + 1][j]
                      << ", stopping iteration at j = " << j << std::endl;
            break;
        }

        // Normalize and assign Q[:,j+1]
        for (int i = 0; i < n; i++)
        {
            Q[i][j + 1] = v[i] / h_next;
        }
    }

    return {H, Q};
}

int main()
{
    // Define the 10x10 matrix A given in the assignment
    std::vector<std::vector<double>> A = {
        {3, 8, 7, 3, 3, 7, 2, 3, 4, 8},
        {5, 4, 1, 6, 9, 8, 3, 7, 1, 9},
        {3, 6, 9, 4, 8, 6, 5, 6, 6, 6},
        {5, 3, 4, 7, 4, 9, 2, 3, 5, 1},
        {4, 4, 2, 1, 7, 4, 2, 2, 4, 5},
        {4, 2, 8, 6, 6, 5, 2, 1, 1, 2},
        {2, 8, 9, 5, 2, 9, 4, 7, 3, 3},
        {9, 3, 2, 2, 7, 3, 4, 8, 7, 7},
        {9, 1, 9, 3, 3, 1, 2, 7, 7, 1},
        {9, 3, 2, 2, 6, 4, 4, 7, 3, 5}};

    // Define the initial vector b (10-dimensional)
    std::vector<double> b = {0.7575, 2.7341, -0.5556, 1.1443, 0.6453,
                             -0.0855, -0.6237, -0.4652, 2.3829, -0.1205};

    // Set the number of iterations
    int iterations = 9;
    auto [H, Q] = arnoldi(b, A, iterations);

    for (unsigned int j = 0; j < Q[0].size(); j++)
    {
        std::cout << "The " << j << "th orthogonal vector is: " << std::endl;
        for (unsigned int i = 0; i < Q.size(); i++)
        {
            std::cout << Q[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "The H matrix is: " << std::endl;

    printMatrix(H);

    // checkOrthogonality(Q);

    // checkArnoldiRelation(A, Q, H, iterations);

    return 0;
}
