#include <stdio.h>  // Standard Input/Output Library, for functions like printf and fprintf.
#include <stdlib.h> // Standard Library, for functions like malloc, free, atoi, and exit.
#include <time.h>   // Time Library, for functions like time (used for seeding random number generator).
#include <math.h>   // Math Library, required for fmaxf (used in ReLU activation).

// Definitions for the MLP dimensions. These are compile-time constants for this implementation.
// BATCH_SIZE, TPB_X, TPB_Y will be passed as command-line arguments at runtime.
#define INPUT_FEATURES 512  // Number of features in the input layer.
#define HIDDEN_NEURONS 2048 // Number of neurons in the hidden layer.
#define OUTPUT_NEURONS 100    // Number of neurons in the output layer (typically 100 for multi-class classification tasks).

// Macro for robust CUDA error checking.
// This macro wraps CUDA API calls and checks their return status.
#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
// Helper inline function for cudaSafeCall.
// If a CUDA error occurs, it prints an error message with the file, line number,
// and error description, then terminates the program.
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error in %s, line %d (%s)\n", file, line, cudaGetErrorString(err));
        // Ensure a clear signal of failure to any calling scripts (e.g., Python) by printing to stderr
        // and exiting with a failure status.
        exit(EXIT_FAILURE);
    }
}

// Rectified Linear Unit (ReLU) activation function.
// This function is executed on the CUDA device (GPU).
// It returns the input if it's positive, and 0 otherwise.
__device__ float relu(float x)
{
    return fmaxf(0.0f, x); // fmaxf returns the larger of two float values.
}

// CUDA kernel for the forward pass of the first layer (Input -> Hidden).
// This kernel computes the weighted sum of inputs and applies the ReLU activation.
// Executed in parallel by multiple threads on the GPU.
__global__ void forward_layer1_naive_kernel(const float *X_global,  // Pointer to the global memory holding the input batch.
                                            const float *W1_global, // Pointer to the global memory holding the weights of the first layer.
                                            const float *b1_global, // Pointer to the global memory holding the biases of the first layer.
                                            float *H_global,        // Pointer to the global memory where the output of the hidden layer will be stored.
                                            int current_batch_size, // The actual batch size for this specific kernel launch (can be smaller than a macro BATCH_SIZE if handling remainders).
                                            int input_features,     // Number of input features, matches INPUT_FEATURES.
                                            int hidden_neurons)
{ // Number of hidden neurons, matches HIDDEN_NEURONS.
    // Calculate the global thread indices for the batch and hidden neuron dimensions.
    // blockIdx & threadIdx are CUDA built-in variables.
    // blockDim is the dimension of a thread block.
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;  // Identifies the sample in the batch.
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x; // Identifies the neuron in the hidden layer.

    // Boundary check: ensure the thread is within the valid range of batch size and hidden neurons.
    // This is important if the number of threads launched exceeds the actual work items.
    if (batch_idx < current_batch_size && hidden_idx < hidden_neurons)
    {
        float sum = 0.0f; // Accumulator for the weighted sum.
        // Compute the dot product of the input features for the current batch sample
        // and the weights corresponding to the current hidden neuron.
        for (int k = 0; k < input_features; ++k)
        {
            // X_global is indexed [batch_idx, k]
            // W1_global is typically indexed [hidden_idx, k] (or [k, hidden_idx] if column-major, here it implies row-major for W1 access per hidden_idx)
            // The current W1_global access W1_global[hidden_idx * input_features + k] implies W1 is stored as (hidden_neurons, input_features)
            sum += X_global[batch_idx * input_features + k] * W1_global[hidden_idx * input_features + k];
        }
        sum += b1_global[hidden_idx]; // Add the bias for the current hidden neuron.
        // Apply the ReLU activation function and store the result in the hidden layer output matrix.
        H_global[batch_idx * hidden_neurons + hidden_idx] = relu(sum);
    }
}

// CUDA kernel for the forward pass of the second layer (Hidden -> Output).
// This kernel computes the weighted sum of hidden layer activations.
// No activation function is applied here, typical for regression output layers.
// Executed in parallel by multiple threads on the GPU.
__global__ void forward_layer2_naive_kernel(const float *H_global,  // Pointer to the global memory holding the hidden layer activations (output of the first layer).
                                            const float *W2_global, // Pointer to the global memory holding the weights of the second layer.
                                            const float *b2_global, // Pointer to the global memory holding the biases of the second layer.
                                            float *Y_global,        // Pointer to the global memory where the final output of the MLP will be stored.
                                            int current_batch_size, // The actual batch size for this kernel launch.
                                            int hidden_neurons,     // Number of hidden neurons, matches HIDDEN_NEURONS.
                                            int output_neurons)
{ // Number of output neurons, matches OUTPUT_NEURONS.
    // Calculate the global thread indices for the batch and output neuron dimensions.
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;  // Identifies the sample in the batch.
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x; // Identifies the neuron in the output layer.

    // Boundary check: ensure the thread is within the valid range of batch size and output neurons.
    if (batch_idx < current_batch_size && output_idx < output_neurons)
    {
        float sum = 0.0f; // Accumulator for the weighted sum.
        // Compute the dot product of the hidden layer activations for the current batch sample
        // and the weights corresponding to the current output neuron.
        for (int k = 0; k < hidden_neurons; ++k)
        {
            // H_global is indexed [batch_idx, k]
            // W2_global is typically indexed [output_idx, k] (or [k, output_idx] if column-major, here it implies row-major for W2 access per output_idx)
            // The current W2_global access W2_global[output_idx * hidden_neurons + k] implies W2 is stored as (output_neurons, hidden_neurons)
            sum += H_global[batch_idx * hidden_neurons + k] * W2_global[output_idx * hidden_neurons + k];
        }
        sum += b2_global[output_idx]; // Add the bias for the current output neuron.
        // Store the result (linear output) in the final output matrix.
        Y_global[batch_idx * output_neurons + output_idx] = sum;
    }
}

// Initializes a matrix with random float values between -1.0 and 1.0.
// This function is executed on the CPU (host).
void initialize_matrix_random(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        // Generates a random float, scales it to [0.0, 1.0], then to [0.0, 2.0], and finally shifts to [-1.0, 1.0].
        matrix[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
    }
}

// Main function of the program.
int main(int argc, char *argv[])
{
    // Validate the number of command-line arguments.
    // Expects program name, batch_size, threads_per_block_x, threads_per_block_y.
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <batch_size> <threads_per_block_x> <threads_per_block_y>\n", argv[0]);
        return EXIT_FAILURE; // Indicate an error in usage.
    }

    // Parse command-line arguments.
    int batch_size_arg = atoi(argv[1]); // Convert batch_size argument from string to integer.
    int tpb_x_arg = atoi(argv[2]);      // Convert threads_per_block_x argument.
    int tpb_y_arg = atoi(argv[3]);      // Convert threads_per_block_y argument.

    // Validate parsed arguments.
    // Batch size and thread block dimensions must be positive.
    if (batch_size_arg <= 0 || tpb_x_arg <= 0 || tpb_y_arg <= 0)
    {
        fprintf(stderr, "Error: batch_size, threads_per_block_x, and threads_per_block_y must be positive.\n");
        return EXIT_FAILURE;
    }
    // Warn if the total number of threads per block is not a multiple of 32 (warp size).
    // Non-multiple of warp size can lead to underutilized warps and suboptimal performance on NVIDIA GPUs.
    if ((tpb_x_arg * tpb_y_arg) % 32 != 0)
    {
        fprintf(stderr, "Warning: total threads per block (tpb_x * tpb_y = %d) is not a multiple of 32. This might lead to suboptimal performance.\n", tpb_x_arg * tpb_y_arg);
    }

    // Seed the random number generator using the current time.
    // This ensures different random numbers are generated on each program run.
    srand(time(NULL));

    // Declare pointers for host (CPU) matrices.
    float *h_X, *h_W1, *h_b1, *h_H, *h_W2, *h_b2, *h_Y;

    // Allocate memory on the host for the MLP matrices.
    // h_X: Input batch (batch_size_arg x INPUT_FEATURES)
    // h_W1: Weights for the first layer (HIDDEN_NEURONS x INPUT_FEATURES)
    // h_b1: Biases for the first layer (HIDDEN_NEURONS x 1)
    // h_H: Hidden layer activations (batch_size_arg x HIDDEN_NEURONS) - also allocated on host though primarily computed on device
    // h_W2: Weights for the second layer (OUTPUT_NEURONS x HIDDEN_NEURONS)
    // h_b2: Biases for the second layer (OUTPUT_NEURONS x 1)
    // h_Y: Final output (batch_size_arg x OUTPUT_NEURONS) - for verification if needed
    h_X = (float *)malloc(batch_size_arg * INPUT_FEATURES * sizeof(float));
    h_W1 = (float *)malloc(HIDDEN_NEURONS * INPUT_FEATURES * sizeof(float));
    h_b1 = (float *)malloc(HIDDEN_NEURONS * sizeof(float));
    h_H = (float *)malloc(batch_size_arg * HIDDEN_NEURONS * sizeof(float)); // Primarily for device use, host copy not strictly needed for GPU timing
    h_W2 = (float *)malloc(OUTPUT_NEURONS * HIDDEN_NEURONS * sizeof(float));
    h_b2 = (float *)malloc(OUTPUT_NEURONS * sizeof(float));
    h_Y = (float *)malloc(batch_size_arg * OUTPUT_NEURONS * sizeof(float)); // For storing results back from GPU if verification is enabled

    // Check if host memory allocations were successful.
    // If any malloc fails, it returns NULL.
    if (!h_X || !h_W1 || !h_b1 || !h_H || !h_W2 || !h_b2 || !h_Y)
    {
        fprintf(stderr, "Host malloc failed!\n");
        // Clean up any memory that was successfully allocated before exiting.
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
    // Input data, weights, and biases are initialized.
    initialize_matrix_random(h_X, batch_size_arg, INPUT_FEATURES);
    initialize_matrix_random(h_W1, HIDDEN_NEURONS, INPUT_FEATURES);
    initialize_matrix_random(h_b1, HIDDEN_NEURONS, 1); // Bias is a vector
    initialize_matrix_random(h_W2, OUTPUT_NEURONS, HIDDEN_NEURONS);
    initialize_matrix_random(h_b2, OUTPUT_NEURONS, 1); // Bias is a vector

    // Declare pointers for device (GPU) matrices.
    float *d_X, *d_W1, *d_b1, *d_H, *d_W2, *d_b2, *d_Y;

    // Allocate memory on the device for the MLP matrices using cudaMalloc.
    // Sizes correspond to their host counterparts.
    cudaSafeCall(cudaMalloc((void **)&d_X, batch_size_arg * INPUT_FEATURES * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_W1, HIDDEN_NEURONS * INPUT_FEATURES * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_b1, HIDDEN_NEURONS * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_H, batch_size_arg * HIDDEN_NEURONS * sizeof(float))); // Intermediate results on device
    cudaSafeCall(cudaMalloc((void **)&d_W2, OUTPUT_NEURONS * HIDDEN_NEURONS * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_b2, OUTPUT_NEURONS * sizeof(float)));
    cudaSafeCall(cudaMalloc((void **)&d_Y, batch_size_arg * OUTPUT_NEURONS * sizeof(float))); // Final results on device

    // Copy data from host memory to device memory.
    // Input (X), weights (W1, W2), and biases (b1, b2) are transferred to the GPU.
    cudaSafeCall(cudaMemcpy(d_X, h_X, batch_size_arg * INPUT_FEATURES * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_W1, h_W1, HIDDEN_NEURONS * INPUT_FEATURES * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_b1, h_b1, HIDDEN_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_W2, h_W2, OUTPUT_NEURONS * HIDDEN_NEURONS * sizeof(float), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_b2, h_b2, OUTPUT_NEURONS * sizeof(float), cudaMemcpyHostToDevice));

    // Configure the number of threads per block for CUDA kernels using parsed arguments.
    // dim3 is a CUDA type for specifying dimensions.
    dim3 threadsPerBlock(tpb_x_arg, tpb_y_arg);

    // Create CUDA events for accurately timing GPU execution.
    // Events are markers in the CUDA stream.
    cudaEvent_t start_event, stop_event;
    cudaSafeCall(cudaEventCreate(&start_event));
    cudaSafeCall(cudaEventCreate(&stop_event));

    // --- Start GPU execution timing ---
    // Record the start event in the default CUDA stream (stream 0).
    cudaSafeCall(cudaEventRecord(start_event, 0));

    // Launch Kernel for Layer 1 (Input -> Hidden).
    // Calculate the number of blocks needed in the grid for layer 1.
    // Grid dimensions are calculated to cover all hidden neurons and batch samples.
    // (total_elements + elements_per_block - 1) / elements_per_block ensures enough blocks are launched.
    dim3 numBlocks1((HIDDEN_NEURONS + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (batch_size_arg + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // Launch the kernel on the GPU.
    forward_layer1_naive_kernel<<<numBlocks1, threadsPerBlock>>>(d_X, d_W1, d_b1, d_H,
                                                                 batch_size_arg, INPUT_FEATURES, HIDDEN_NEURONS);
    // Check for any errors launched by the kernel or during its execution.
    // cudaGetLastError() checks for asynchronous errors from previous CUDA calls in the same stream.
    cudaSafeCall(cudaGetLastError());

    // Launch Kernel for Layer 2 (Hidden -> Output).
    // Calculate the number of blocks needed in the grid for layer 2.
    // Grid dimensions are calculated to cover all output neurons and batch samples.
    dim3 numBlocks2((OUTPUT_NEURONS + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (batch_size_arg + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // Launch the kernel on the GPU.
    forward_layer2_naive_kernel<<<numBlocks2, threadsPerBlock>>>(d_H, d_W2, d_b2, d_Y,
                                                                 batch_size_arg, HIDDEN_NEURONS, OUTPUT_NEURONS);
    // Check for any errors from the second kernel launch.
    cudaSafeCall(cudaGetLastError());

    // --- End GPU execution timing ---
    // Record the stop event. This is also an asynchronous call.
    cudaSafeCall(cudaEventRecord(stop_event, 0));
    // Wait for the stop_event to complete. This is a blocking call that ensures all preceding
    // GPU operations in the stream (including the kernels and the stop_event recording) are finished.
    // Essential for accurate timing.
    cudaSafeCall(cudaEventSynchronize(stop_event));

    float milliseconds = 0; // Variable to store the elapsed time.
    // Calculate the time elapsed between start_event and stop_event.
    cudaSafeCall(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

    /* Optional: Copy output Y from device to host for verification.
       This section is commented out as it's not part of the GPU execution time measurement.
    cudaSafeCall(cudaMemcpy(h_Y, d_Y, batch_size_arg * OUTPUT_NEURONS * sizeof(float), cudaMemcpyDeviceToHost));

    // Print the results (output for each sample in the batch).
    // Assumes OUTPUT_NEURONS is 1 for this printing format.
    printf("Output Y (first %d elements if batch is large):\n", batch_size_arg > 10 ? 10 : batch_size_arg);
    for (int i = 0; i < (batch_size_arg > 10 ? 10 : batch_size_arg) ; ++i) { // Print first few elements
        printf("%f\n", h_Y[i * OUTPUT_NEURONS]); // Assuming OUTPUT_NEURONS is 1 or taking the first output feature.
    }
    */

    // Output results in CSV format for easy parsing by scripts (e.g., Python).
    // Format: BatchSize,TPB_X,TPB_Y,TotalThreadsPerBlock,GPUTime_ms
    printf("%d,%d,%d,%d,%.4f\n",
           batch_size_arg,
           tpb_x_arg,
           tpb_y_arg,
           tpb_x_arg * tpb_y_arg, // Total threads per block
           milliseconds);

    // Clean up CUDA events.
    cudaSafeCall(cudaEventDestroy(start_event));
    cudaSafeCall(cudaEventDestroy(stop_event));

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