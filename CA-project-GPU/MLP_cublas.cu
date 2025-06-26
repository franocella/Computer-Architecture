#include <stdio.h>     // Standard Input/Output Library, for functions like printf and fprintf.
#include <stdlib.h>    // Standard Library, for functions like malloc, free, atoi, and exit.
#include <time.h>      // Time Library, for functions like time (used for seeding random number generator).
#include <math.h>      // Math Library, required for fmaxf (used in ReLU activation).
#include <cublas_v2.h> // cuBLAS Library, for Basic Linear Algebra Subprograms on NVIDIA GPUs.

// Define dimensions for the Multi-Layer Perceptron (MLP).
// These are compile-time constants. Dimensions are increased to better observe cuBLAS performance.
#define INPUT_FEATURES_DIM 512  // Number of features in the input layer.
#define HIDDEN_NEURONS_DIM 2048 // Number of neurons in the hidden layer.
#define OUTPUT_NEURONS_DIM 100    // Number of neurons in the output layer (typically 100 for classification tasks).

// Macro for robust CUDA error checking.
// Wraps CUDA API calls and checks their return status.
#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
// Helper inline function for cudaSafeCall.
// If a CUDA error occurs, it prints an error message and terminates the program.
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error in %s, line %d (%s)\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Macro for robust cuBLAS error checking.
// Wraps cuBLAS API calls and checks their return status.
#define cublasSafeCall(err) __cublasSafeCall(err, __FILE__, __LINE__)
// Helper inline function for cublasSafeCall.
// If a cuBLAS error occurs, it prints an error message and terminates the program.
inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line)
{
    if (CUBLAS_STATUS_SUCCESS != err)
    {
        fprintf(stderr, "cuBLAS error in %s, line %d (Code: %d)\n", file, line, err);
        exit(EXIT_FAILURE);
    }
}

// CUDA kernel to add a bias vector to a matrix and apply the ReLU activation function element-wise.
// The matrix_out is modified in-place: matrix_out = ReLU(matrix_out + bias_vector_broadcasted).
// Each row of matrix_out has the same bias vector added to it.
__global__ void add_bias_relu_kernel(float *matrix_out, // Pointer to the input/output matrix (num_rows x num_cols) on the device.
                                     const float *bias, // Pointer to the bias vector (1 x num_cols) on the device.
                                     int num_rows,      // Number of rows in matrix_out.
                                     int num_cols)
{ // Number of columns in matrix_out (and size of bias vector).
    // Calculate global thread indices for row and column.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure thread is within matrix dimensions.
    if (row < num_rows && col < num_cols)
    {
        int index = row * num_cols + col; // Linear index for row-major matrix.
        // Add bias and apply ReLU: matrix_out[index] = max(0, matrix_out[index] + bias[col]).
        matrix_out[index] = fmaxf(0.0f, matrix_out[index] + bias[col]);
    }
}

// CUDA kernel to add a bias vector to a matrix element-wise.
// Used for the output layer where ReLU activation is typically not applied for regression.
// The matrix_out is modified in-place: matrix_out = matrix_out + bias_vector_broadcasted.
__global__ void add_bias_kernel(float *matrix_out, // Pointer to the input/output matrix (num_rows x num_cols) on the device.
                                const float *bias, // Pointer to the bias vector (1 x num_cols) on the device.
                                int num_rows,      // Number of rows in matrix_out.
                                int num_cols)
{ // Number of columns in matrix_out (and size of bias vector).
    // Calculate global thread indices for row and column.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure thread is within matrix dimensions.
    if (row < num_rows && col < num_cols)
    {
        int index = row * num_cols + col; // Linear index for row-major matrix.
        // Add bias: matrix_out[index] = matrix_out[index] + bias[col].
        matrix_out[index] = matrix_out[index] + bias[col];
    }
}

// Initializes a matrix with random float values.
// Values are initialized in a smaller range [-0.1, 0.1] for potentially better numerical stability during training.
// This function is executed on the CPU (host).
void initialize_matrix_random(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        // Generates a random float, scales it to [0.0, 1.0], then to [0.0, 0.2], and finally shifts to [-0.1, 0.1].
        matrix[i] = (float)rand() / (float)RAND_MAX * 0.2f - 0.1f;
    }
}

// Main function of the program.
int main(int argc, char *argv[])
{
    // Validate the number of command-line arguments.
    // Expects: program_name, batch_size, threads_per_block_x (for bias/ReLU kernels), threads_per_block_y (for bias/ReLU kernels).
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <batch_size> <threads_per_block_x_for_bias_relu> <threads_per_block_y_for_bias_relu>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse command-line arguments.
    int batch_size_arg = atoi(argv[1]);  // Batch size for processing.
    int tpb_x_bias_relu = atoi(argv[2]); // Threads per block in X dimension for bias/ReLU kernels.
    int tpb_y_bias_relu = atoi(argv[3]); // Threads per block in Y dimension for bias/ReLU kernels.

    // Validate parsed arguments.
    if (batch_size_arg <= 0 || tpb_x_bias_relu <= 0 || tpb_y_bias_relu <= 0)
    {
        fprintf(stderr, "Error: batch_size and threads_per_block for bias/ReLU kernels must be positive.\n");
        return EXIT_FAILURE;
    }
    // Warn if total threads per block for bias/ReLU kernels is not a multiple of 32 (warp size).
    // This can lead to suboptimal performance due to underutilized warps.
    if ((tpb_x_bias_relu * tpb_y_bias_relu) % 32 != 0)
    {
        fprintf(stderr, "Warning: total threads per block for bias/ReLU kernels (%d * %d = %d) is not a multiple of 32. This might lead to suboptimal performance.\n",
                tpb_x_bias_relu, tpb_y_bias_relu, tpb_x_bias_relu * tpb_y_bias_relu);
    }

    // Seed the random number generator.
    srand(time(NULL));

    // Declare pointers for host (CPU) matrices.
    float *h_X, *h_W1, *h_b1, *h_H, *h_W2, *h_b2, *h_Y;

    // Allocate memory on the host for MLP matrices.
    // Matrices are stored in row-major order.
    // h_X: Input batch (batch_size_arg x INPUT_FEATURES_DIM)
    // h_W1: Weights for Layer 1 (INPUT_FEATURES_DIM x HIDDEN_NEURONS_DIM)
    // h_b1: Biases for Layer 1 (1 x HIDDEN_NEURONS_DIM)
    // h_H: Hidden layer activations (batch_size_arg x HIDDEN_NEURONS_DIM) - stores X*W1, then ReLU(X*W1 + b1)
    // h_W2: Weights for Layer 2 (HIDDEN_NEURONS_DIM x OUTPUT_NEURONS_DIM)
    // h_b2: Biases for Layer 2 (1 x OUTPUT_NEURONS_DIM)
    // h_Y: Final output (batch_size_arg x OUTPUT_NEURONS_DIM)
    h_X = (float *)malloc(batch_size_arg * INPUT_FEATURES_DIM * sizeof(float));
    h_W1 = (float *)malloc(INPUT_FEATURES_DIM * HIDDEN_NEURONS_DIM * sizeof(float)); // W1: (Input Features x Hidden Neurons)
    h_b1 = (float *)malloc(HIDDEN_NEURONS_DIM * sizeof(float));                      // b1: (1 x Hidden Neurons)
    h_H = (float *)malloc(batch_size_arg * HIDDEN_NEURONS_DIM * sizeof(float));      // Intermediate for H = X*W1
    h_W2 = (float *)malloc(HIDDEN_NEURONS_DIM * OUTPUT_NEURONS_DIM * sizeof(float)); // W2: (Hidden Neurons x Output Neurons)
    h_b2 = (float *)malloc(OUTPUT_NEURONS_DIM * sizeof(float));                      // b2: (1 x Output Neurons)
    h_Y = (float *)malloc(batch_size_arg * OUTPUT_NEURONS_DIM * sizeof(float));

    // Check for host memory allocation failures.
    if (!h_X || !h_W1 || !h_b1 || !h_H || !h_W2 || !h_b2 || !h_Y)
    {
        fprintf(stderr, "Host malloc failed!\n");
        // Free any successfully allocated memory before exiting.
        free(h_X);
        free(h_W1);
        free(h_b1);
        free(h_H);
        free(h_W2);
        free(h_b2);
        free(h_Y);
        return EXIT_FAILURE;
    }

    // Initialize host matrices with random values.
    initialize_matrix_random(h_X, batch_size_arg, INPUT_FEATURES_DIM);
    initialize_matrix_random(h_W1, INPUT_FEATURES_DIM, HIDDEN_NEURONS_DIM);
    initialize_matrix_random(h_b1, 1, HIDDEN_NEURONS_DIM); // Bias is a row vector.
    initialize_matrix_random(h_W2, HIDDEN_NEURONS_DIM, OUTPUT_NEURONS_DIM);
    initialize_matrix_random(h_b2, 1, OUTPUT_NEURONS_DIM); // Bias is a row vector.

    // Declare pointers for device (GPU) matrices.
    float *d_X, *d_W1, *d_b1, *d_H, *d_W2, *d_b2, *d_Y;

    // Allocate memory on the device for MLP matrices.
    cudaSafeCall(cudaMalloc((void **)&d_X, batch_size_arg * INPUT_FEATURES_DIM * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_W1, INPUT_FEATURES_DIM * HIDDEN_NEURONS_DIM * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_b1, HIDDEN_NEURONS_DIM * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_H, batch_size_arg * HIDDEN_NEURONS_DIM * sizeof(float))); // d_H will store the result of X*W1, then activation.
    cudaSafeCall(cudaMalloc((void **)&d_W2, HIDDEN_NEURONS_DIM * OUTPUT_NEURONS_DIM * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_b2, OUTPUT_NEURONS_DIM * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_Y, batch_size_arg * OUTPUT_NEURONS_DIM * sizeof(float))); // d_Y will store the result of H*W2, then final output.

    // Copy data from host memory to device memory.
    cudaSafeCall(cudaMemcpy(d_X, h_X, batch_size_arg * INPUT_FEATURES_DIM * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_W1, h_W1, INPUT_FEATURES_DIM * HIDDEN_NEURONS_DIM * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_b1, h_b1, HIDDEN_NEURONS_DIM * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_W2, h_W2, HIDDEN_NEURONS_DIM * OUTPUT_NEURONS_DIM * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_b2, h_b2, OUTPUT_NEURONS_DIM * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize cuBLAS library context.
    cublasHandle_t cublasH;
    cublasSafeCall(cublasCreate(&cublasH));

    // Define scaling factors for cuBLAS SGEMM: C = alpha*A*B + beta*C.
    float alpha = 1.0f; // Multiplier for A*B.
    float beta = 0.0f;  // Multiplier for existing C. If beta=0, C is overwritten and not read.

    // Create CUDA events for timing GPU execution.
    cudaEvent_t start_event, stop_event;
    cudaSafeCall(cudaEventCreate(&start_event));
    cudaSafeCall(cudaEventCreate(&stop_event));

    printf("Starting MLP computation with cuBLAS and custom bias/activation kernels.\n");
    printf("Dimensions: Input=%d, Hidden=%d, Output=%d, BatchSize=%d\n",
           INPUT_FEATURES_DIM, HIDDEN_NEURONS_DIM, OUTPUT_NEURONS_DIM, batch_size_arg);

    // --- Start GPU execution timing ---
    cudaSafeCall(cudaEventRecord(start_event, 0));

    // Layer 1: Compute H_intermediate = X * W1 using cuBLAS SGEMM.
    // X is (batch_size_arg x INPUT_FEATURES_DIM)         [mat A in math: m x k]
    // W1 is (INPUT_FEATURES_DIM x HIDDEN_NEURONS_DIM)    [mat B in math: k x n]
    // H_intermediate is (batch_size_arg x HIDDEN_NEURONS_DIM) [mat C in math: m x n]
    //
    // For row-major matrices A(m,k), B(k,n), C(m,n) where C = A*B,
    // use cublasSgemm as: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //                                n_cols_B, m_rows_A, k_cols_A_rows_B,
    //                                alpha, B_ptr, n_cols_B, A_ptr, k_cols_A_rows_B,
    //                                beta, C_ptr, n_cols_B)
    // Here: A_math = X, B_math = W1, C_math = d_H
    // m_math = batch_size_arg
    // k_math = INPUT_FEATURES_DIM
    // n_math = HIDDEN_NEURONS_DIM
    cublasSafeCall(cublasSgemm(cublasH,
                               CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose for W1 and X
                               HIDDEN_NEURONS_DIM,        // n (cols of W1, cols of d_H)
                               batch_size_arg,            // m (rows of X, rows of d_H)
                               INPUT_FEATURES_DIM,        // k (cols of X / rows of W1)
                               &alpha,                    // Scalar alpha
                               d_W1, HIDDEN_NEURONS_DIM,  // B_math (W1) and its leading dimension (num_cols of W1)
                               d_X, INPUT_FEATURES_DIM,   // A_math (X) and its leading dimension (num_cols of X)
                               &beta,                     // Scalar beta
                               d_H, HIDDEN_NEURONS_DIM)); // C_math (d_H) and its leading dimension (num_cols of d_H)
    cudaSafeCall(cudaGetLastError());                     // Check for errors after cuBLAS call.

    // Add bias b1 to d_H and apply ReLU activation: d_H = ReLU( (X*W1) + b1 )
    // d_H currently holds the result of X*W1. This operation is done in-place.
    dim3 threadsPerBlockBiasRelu1(tpb_x_bias_relu, tpb_y_bias_relu);
    dim3 numBlocksBiasRelu1((HIDDEN_NEURONS_DIM + threadsPerBlockBiasRelu1.x - 1) / threadsPerBlockBiasRelu1.x,
                            (batch_size_arg + threadsPerBlockBiasRelu1.y - 1) / threadsPerBlockBiasRelu1.y);
    add_bias_relu_kernel<<<numBlocksBiasRelu1, threadsPerBlockBiasRelu1>>>(d_H, d_b1, batch_size_arg, HIDDEN_NEURONS_DIM);
    cudaSafeCall(cudaGetLastError()); // Check for errors after kernel launch.

    // Layer 2: Compute Y_intermediate = H * W2 using cuBLAS SGEMM.
    // H is (batch_size_arg x HIDDEN_NEURONS_DIM)          [mat A in math: m x k] (result from previous layer)
    // W2 is (HIDDEN_NEURONS_DIM x OUTPUT_NEURONS_DIM)    [mat B in math: k x n]
    // Y_intermediate is (batch_size_arg x OUTPUT_NEURONS_DIM) [mat C in math: m x n]
    //
    // Using the same row-major convention for cuBLAS SGEMM:
    // Here: A_math = d_H, B_math = W2, C_math = d_Y
    // m_math = batch_size_arg
    // k_math = HIDDEN_NEURONS_DIM
    // n_math = OUTPUT_NEURONS_DIM
    cublasSafeCall(cublasSgemm(cublasH,
                               CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose for W2 and d_H
                               OUTPUT_NEURONS_DIM,        // n (cols of W2, cols of d_Y)
                               batch_size_arg,            // m (rows of d_H, rows of d_Y)
                               HIDDEN_NEURONS_DIM,        // k (cols of d_H / rows of W2)
                               &alpha,                    // Scalar alpha
                               d_W2, OUTPUT_NEURONS_DIM,  // B_math (W2) and its leading dimension (num_cols of W2)
                               d_H, HIDDEN_NEURONS_DIM,   // A_math (d_H) and its leading dimension (num_cols of d_H)
                               &beta,                     // Scalar beta
                               d_Y, OUTPUT_NEURONS_DIM)); // C_math (d_Y) and its leading dimension (num_cols of d_Y)
    cudaSafeCall(cudaGetLastError());                     // Check for errors after cuBLAS call.

    // Add bias b2 to d_Y: d_Y = (H*W2) + b2. No ReLU for the output layer in this regression example.
    // This operation is done in-place on d_Y.
    dim3 threadsPerBlockBias2(tpb_x_bias_relu, tpb_y_bias_relu); // Reusing thread block configuration from previous bias kernel.
    dim3 numBlocksBias2((OUTPUT_NEURONS_DIM + threadsPerBlockBias2.x - 1) / threadsPerBlockBias2.x,
                        (batch_size_arg + threadsPerBlockBias2.y - 1) / threadsPerBlockBias2.y);
    add_bias_kernel<<<numBlocksBias2, threadsPerBlockBias2>>>(d_Y, d_b2, batch_size_arg, OUTPUT_NEURONS_DIM);
    cudaSafeCall(cudaGetLastError()); // Check for errors after kernel launch.

    // --- End GPU execution timing ---
    cudaSafeCall(cudaEventRecord(stop_event, 0));
    // Wait for all preceding GPU operations in the stream to complete. Essential for accurate timing.
    cudaSafeCall(cudaEventSynchronize(stop_event));

    float milliseconds = 0; // Variable to store the elapsed GPU execution time.
    // Calculate the time elapsed between the start and stop events.
    cudaSafeCall(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    // Optional: Copy final output Y from device to host for verification.
    // This is not included in the primary GPU timing.
    /*
    cudaSafeCall(cudaMemcpy(h_Y, d_Y, batch_size_arg * OUTPUT_NEURONS_DIM * sizeof(float), cudaMemcpyDeviceToHost));
    // Print the output of the first sample for a quick check (if OUTPUT_NEURONS_DIM is 1).
    if (batch_size_arg > 0 && OUTPUT_NEURONS_DIM > 0) {
         printf("Sample 0 Output: %f\n", h_Y[0]);
    }
    */

    // Output performance results in CSV format for easy parsing.
    // BatchSize,TPB_X_BiasKernel,TPB_Y_BiasKernel,TotalThreads_BiasKernel,GPUTime_ms
    printf("%d,%d,%d,%d,%.4f\n",
           batch_size_arg,
           tpb_x_bias_relu,
           tpb_y_bias_relu,
           tpb_x_bias_relu * tpb_y_bias_relu, // Total threads used per block for bias/activation kernels.
           milliseconds);

    // Cleanup resources.
    // Destroy CUDA events.
    cudaSafeCall(cudaEventDestroy(start_event));
    cudaSafeCall(cudaEventDestroy(stop_event));
    // Destroy cuBLAS handle.
    cublasSafeCall(cublasDestroy(cublasH));

    // Free allocated host memory.
    free(h_X);
    free(h_W1);
    free(h_b1);
    free(h_H);
    free(h_W2);
    free(h_b2);
    free(h_Y);
    // Free allocated device memory.
    cudaSafeCall(cudaFree(d_X));
    cudaSafeCall(cudaFree(d_W1));
    cudaSafeCall(cudaFree(d_b1));
    cudaSafeCall(cudaFree(d_H));
    cudaSafeCall(cudaFree(d_W2));
    cudaSafeCall(cudaFree(d_b2));
    cudaSafeCall(cudaFree(d_Y));

    return EXIT_SUCCESS; // Indicate successful program execution.
}