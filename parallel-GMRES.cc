/**
 * @file parallel-GMRES.cpp
 * @brief Parallel implementation of GMRES algorithm
 * @author Yujin Han
 * @date 2025-03-25
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <mpi.h>

// Compute the Euclidean (L2) norm of a vector.
double norm2(const std::vector<double> &v)
{
    int irank;
    int isize;
    MPI_Comm_rank(MPI_COMM_WORLD, &irank);
    MPI_Comm_size(MPI_COMM_WORLD, &isize);
    int n = v.size();
    int n_local = n / isize;
    int my_n = n_local;
    if (irank == isize - 1)
    {
        my_n += n % isize;
    }
    int my_start = irank * n_local;
    int my_end = my_start + my_n;
    double sum = 0.0;
    for (int i = my_start; i < my_end; i++)
    {
        sum += v[i] * v[i];
    }
    double sum_all;
    MPI_Allreduce(&sum, &sum_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return std::sqrt(sum_all);
}

// Multiply matrix A (n x n) with vector v (length n); return result.
std::vector<double> matVecProduct(const std::vector<std::vector<double>> &A, const std::vector<double> &v)
{
    int irank;
    int isize;
    MPI_Comm_rank(MPI_COMM_WORLD, &irank);
    MPI_Comm_size(MPI_COMM_WORLD, &isize);
    int n = A.size();
    int n_local = n / isize;
    int my_n = n_local;
    if (irank == isize - 1)
    {
        my_n += n % isize;
    }
    int my_start = irank * n_local;

    std::vector<double> w(my_n, 0.0);
    for (int i = 0; i < my_n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            w[i] += A[i + my_start][j] * v[j];
        }
    }
    std::vector<double> w_all(n, 0.0);
    MPI_Allgather(&w[0], my_n, MPI_DOUBLE, &w_all[0], my_n, MPI_DOUBLE, MPI_COMM_WORLD);
    return w_all;
}

// Generate the special tridiagonal matrix A of size n x n:
// Diagonal entries: -4, sub- and super-diagonals: 1.
std::vector<std::vector<double>> generateMatrixA(int n)
{
    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++)
    {
        A[i][i] = -4.0;
        if (i > 0)
        {
            A[i][i - 1] = 1.0;
        }
        if (i < n - 1)
        {
            A[i][i + 1] = 1.0;
        }
    }
    return A;
}

// Generate the vector b of length n, with b[i] = (i+1)/n for i=0,...,n-1.
std::vector<double> generateVectorb(int n)
{
    std::vector<double> b(n, 0.0);
    for (int i = 0; i < n; i++)
    {
        b[i] = double(i + 1.0) / double(n);
    }
    return b;
}

// Apply a Givens rotation to (dx, dy) using cosine cs and sine sn.
void applyGivensRotation(const double &cs, const double &sn, double &dx, double &dy)
{
    double temp = cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
}

// Generate Givens rotation parameters (cs and sn) to eliminate dy in (dx, dy).
void generateGivensRotation(const double &dx, const double &dy, double &cs, double &sn)
{
    if (std::fabs(dy) < 1e-14)
    {
        cs = 1.0;
        sn = 0.0;
    }
    else
    {
        double r = std::sqrt(dx * dx + dy * dy);
        cs = dx / r;
        sn = dy / r;
    }
}

// Gram-Schmidt orthogonalization
void gramSchmidt(const std::vector<std::vector<double>> &V, std::vector<double> &w, std::vector<std::vector<double>> &H, int k)
{
    int irank;
    int isize;
    MPI_Comm_rank(MPI_COMM_WORLD, &irank);
    MPI_Comm_size(MPI_COMM_WORLD, &isize);
    int n = V.size();
    int n_local = n / isize;
    int my_n = n_local;
    if (irank == isize - 1)
    {
        my_n += n % isize;
    }
    int my_start = irank * n_local;
    int my_end = my_start + my_n;

    for (int i = 0; i <= k; i++)
    {
        double dot = 0.0;
        for (int j = my_start; j < my_end; j++)
        {
            dot += V[j][i] * w[j];
        }
        double dot_all;
        MPI_Allreduce(&dot, &dot_all, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        H[i][k] = dot_all;

        std::vector<double> w_local(my_n, 0.0);
        for (int j = 0; j < my_n; j++)
        {
            w_local[j] = w[j + my_start];
        }
        for (int j = my_start; j < my_end; j++)
        {
            w_local[j - my_start] -= dot_all * V[j][i];
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(&w_local[0], my_n, MPI_DOUBLE, &w[0], my_n, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    H[k + 1][k] = norm2(w);
}

/**
 * @brief Parallel GMRES implementation.
 *  The main difference between the parallel and serial GMRES is that the parallel GMRES is the parallelization of the matrix-vector product,
 *  vector norm, and the Gram-Schmidt orthogonalization.
 * @param A The n x n matrix.
 * @param b The right-hand side vector (length n).
 * @param m The number of GMRES iterations (m = n/2 in the experiments).
 * @param tol Tolerance for the stopping criterion.
 * @return A pair: (x, residualHistory), where:
 *         - x is the final approximate solution,
 *         - residualHistory is a vector of residual norms at each iteration.
 */
std::pair<std::vector<double>, std::vector<double>> gmres(
    const std::vector<std::vector<double>> &A,
    const std::vector<double> &b,
    const int m,
    const double tol = 1e-8)
{
    int n = b.size();
    // Initial guess x0 = 0
    std::vector<double> x(n, 0.0);

    // Initial residual: r0 = b (since x0 = 0), and beta = ||r0||
    std::vector<double> r0 = b;
    double beta = norm2(r0);
    std::vector<double> residualHistory;
    residualHistory.push_back(beta);
    if (beta < tol)
    {
        return {x, residualHistory};
    }

    // V will store the orthonormal basis (n x (m+1))
    std::vector<std::vector<double>> V(n, std::vector<double>(m + 1, 0.0));
    // Set v1 = r0 / beta (placing b as the first COLUMN of V)
    for (int i = 0; i < n; i++)
    {
        V[i][0] = r0[i] / beta;
    }

    // H is the (m+1) x m upper Hessenberg matrix.
    std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));

    // Arrays for storing Givens rotation parameters.
    std::vector<double> cs(m, 0.0), sn(m, 0.0);

    // g is the right-hand side for the least squares problem.
    std::vector<double> g(m + 1, 0.0);
    g[0] = beta;

    int k;
    for (k = 0; k < m; k++)
    {
        // Compute w = A * v_k, where v_k is the k-th column of V.
        std::vector<double> v_k(n, 0.0);
        for (int i = 0; i < n; i++)
        {
            v_k[i] = V[i][k];
        }
        std::vector<double> w = matVecProduct(A, v_k);

        // Modified Gram-Schmidt orthogonalization against V[:,0..k]
        gramSchmidt(V, w, H, k);

        H[k + 1][k] = norm2(w);

        // if (H[k + 1][k] < tol * 2e-7)
        // {
        //     break;
        // }

        // Normalize w to get the new basis vector v_{k+1}
        for (int i = 0; i < n; i++)
        {
            V[i][k + 1] = w[i] / H[k + 1][k];
        }

        // Apply previously computed Givens rotations to H[0..k+1][k]
        for (int i = 0; i < k; i++)
        {
            double temp = cs[i] * H[i][k] + sn[i] * H[i + 1][k];
            H[i + 1][k] = -sn[i] * H[i][k] + cs[i] * H[i + 1][k];
            H[i][k] = temp;
        }

        // Compute new Givens rotation to eliminate H[k+1][k]
        double r = std::sqrt(H[k][k] * H[k][k] + H[k + 1][k] * H[k + 1][k]);
        cs[k] = H[k][k] / r;
        sn[k] = H[k + 1][k] / r;
        applyGivensRotation(cs[k], sn[k], H[k][k], H[k + 1][k]);
        applyGivensRotation(cs[k], sn[k], g[k], g[k + 1]);

        double res = std::fabs(g[k + 1]);
        residualHistory.push_back(res);

        // Absolute residual check
        if (res < tol * 2e-7)
            break;
    }

    // Solve the upper triangular system for y via back substitution.
    int actualIters = k; // k iterations were performed.
    std::vector<double> y(actualIters, 0.0);
    for (int i = actualIters - 1; i >= 0; i--)
    {
        double sum = g[i];
        for (int j = i + 1; j < actualIters; j++)
        {
            sum -= H[i][j] * y[j];
        }
        y[i] = sum / H[i][i];
    }

    // Compute the final solution: x = x0 + V(:,0..actualIters) * y (x0 = 0).
    for (int j = 0; j < actualIters; j++)
    {
        for (int i = 0; i < n; i++)
        {
            x[i] += V[i][j] * y[j];
        }
    }

    return {x, residualHistory};
}

int main()
{
    // Initialize MPI
    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define the system sizes.
    std::vector<int> sizes = {8, 16, 32, 64, 128, 256};

    // Open a file to output residual histories for plotting.
    std::ofstream ofs("gmres_residuals_paralell.txt");

    for (int n : sizes)
    {
        std::cout << "------------------------------" << std::endl;
        std::cout << "Dimension size n = " << n << std::endl;

        // Generate the matrix A and vector b.
        std::vector<std::vector<double>> A = generateMatrixA(n);
        std::vector<double> b = generateVectorb(n);

        // Set the number of GMRES iterations m = n/2.
        int m = n / 2;

        // Run GMRES.
        auto [x, residualHistory] = gmres(A, b, m, 1e-8);

        // Output the final solution (optional).
        if (rank == 0)
        {
            std::cout << "Final solution x:" << std::endl;
            for (double xi : x)
            {
                std::cout << xi << std::endl;
            }
            std::cout << std::endl;
        }

        // Output the normalized residual history.
        if (rank == 0)
        {
            std::cout << "Normalized residual history (||r_k||/||b||):" << std::endl;
            ofs << "n = " << n << std::endl;
            for (size_t i = 1; i < residualHistory.size(); i++)
            {
                double normalized = residualHistory[i] / residualHistory[0];
                std::cout << "iter " << i << ": " << normalized << std::endl;
                ofs << i << " " << normalized << std::endl;
            }
            ofs << std::endl;
            std::cout << "Residual data written to gmres_residuals_paralell.txt." << std::endl;
        }
    }

    ofs.close();

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
