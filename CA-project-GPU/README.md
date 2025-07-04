# CUDA-Accelerated Multi-Layer Perceptron (MLP): Naive vs. cuBLAS Implementations

## Purpose

This project presents and compares two CUDA C++ implementations of a Multi-Layer Perceptron (MLP) for feedforward inference:
1.  A **"naive" implementation** (`MLP_naive.cu`): Utilizes custom CUDA kernels for matrix multiplication and activation functions.
2.  A **cuBLAS-optimized implementation** (`MLP_cublas.cu`): Leverages the highly optimized `cublasSgemm` routine from the NVIDIA cuBLAS library for matrix multiplications, combined with custom CUDA kernels for bias addition and activation functions.

The primary goal is to demonstrate GPU acceleration for neural network computations and to quantify the performance benefits of using optimized libraries like cuBLAS over custom-written kernels for common linear algebra operations. Both implementations perform inference on randomly initialized data and measure GPU execution time. The project includes Python scripts for execution management (`execute.py`) and results analysis (`analysis.ipynb`).

## Theory

### Multi-Layer Perceptron (MLP)

A Multi-Layer Perceptron (MLP) is a class of feedforward artificial neural network. This project implements a simple MLP with one input layer, one hidden layer, and one output layer. The forward propagation process is as follows:

1.  **Hidden Layer Calculation:**
    $Z_1 = X \cdot W_1$
    $H = \text{ReLU}(Z_1 + b_1)$
    Where:
    * $X$ is the input matrix (batch_size x input_features).
    * $W_1$ are the weights of the first layer (input_features x hidden_neurons for cuBLAS, hidden_neurons x input_features for naive kernel's direct indexing).
    * $b_1$ is the bias vector for the hidden layer.
    * ReLU (Rectified Linear Unit) is the activation function: $\text{ReLU}(x) = \max(0, x)$.
    * $H$ is the output of the hidden layer (batch_size x hidden_neurons).

2.  **Output Layer Calculation:**
    $Z_2 = H \cdot W_2$
    $Y = Z_2 + b_2$
    Where:
    * $H$ is the activation from the hidden layer.
    * $W_2$ are the weights of the second layer (hidden_neurons x output_neurons for cuBLAS, output_neurons x hidden_neurons for naive kernel's direct indexing).
    * $b_2$ is the bias vector for the output layer.
    * $Y$ is the final output of the network (batch_size x output_neurons). No activation function is applied to the output layer, typical for regression tasks.

The network dimensions (`INPUT_FEATURES`, `HIDDEN_NEURONS`, `OUTPUT_NEURONS`) are defined as compile-time constants in the source files.

### cuBLAS for Optimization

**cuBLAS** is a GPU-accelerated version of the Basic Linear Algebra Subprograms (BLAS) library. It provides highly optimized implementations of common linear algebra operations. In this project, `cublasSgemm` (Single-precision General Matrix Multiply) is used for the $X \cdot W_1$ and $H \cdot W_2$ operations. This is expected to yield significant performance improvements over manually implemented kernels, especially for larger matrices, as NVIDIA heavily optimizes these routines for their GPU architectures.

For the cuBLAS implementation, matrix multiplications are performed first, followed by separate custom CUDA kernels to add biases and apply the ReLU activation function.

Matrices are stored in row-major order. The `cublasSgemm` function is called in a way that accommodates row-major storage for $C = A \cdot B$. Specifically, if $A$ (e.g., $X$), $B$ (e.g., $W_1$), and $C$ (e.g., $d\_H$) are row-major, the operation $C_{m \times n} = A_{m \times k} \cdot B_{k \times n}$ is achieved by calling `cublasSgemm` with matrix $B$ as the first matrix argument and $A$ as the second, adjusting dimensions accordingly (effectively computing $(C^T = B^T \cdot A^T)^T$).

## Project Structure

The project is organized in a flat directory structure:
```plaintext
.
├── .gitignore
├── analysis.ipynb
├── execute.py
├── MLP_cublas.cu
├── MLP_naive.cu
├── README.md
└── requirements.txt
```

* **MLP_naive.cu**: C++/CUDA source code for the naive MLP implementation.
* **MLP_cublas.cu**: C++/CUDA source code for the cuBLAS-optimized MLP implementation.
* **execute.py**: Python script likely used to automate the compilation and/or execution of the CUDA applications, potentially running benchmarks over various parameters and collecting results.
* **analysis.ipynb**: Jupyter Notebook for analyzing and visualizing the performance data generated by the MLP applications (likely from their CSV output).
* **requirements.txt**: Lists the Python dependencies required for `execute.py` and `analysis.ipynb`.
* **README.md**: This file.
* **.gitignore**: Specifies intentionally untracked files that Git should ignore.

## Requirements

### Hardware:
* NVIDIA GPU with CUDA support (Compute Capability 3.5 or higher recommended).

### Software:
* **CUDA Toolkit**: (e.g., 11.x, 12.x). Provides `nvcc` compiler, CUDA libraries (including cuBLAS), and headers.
* **C++ Compiler**: (e.g., GCC, Clang, MSVC) Compatible with your CUDA Toolkit version.
* **CMake**: (e.g., 3.10+) Build system generator (recommended for easier compilation of CUDA codes).
* **Python**: (e.g., 3.7+) For running `execute.py` and `analysis.ipynb`.
* **Python Libraries**: Specified in `requirements.txt`. Typically include:
    * `pandas` (for data manipulation of CSV outputs)
    * `matplotlib` / `seaborn` (for plotting in `analysis.ipynb`)
    * `numpy` (for numerical operations)
    * `jupyter` (to run `analysis.ipynb`)

### CUDA Toolkit Installation
1.  Download the CUDA Toolkit from the [NVIDIA CUDA Downloads page](https://developer.nvidia.com/cuda-downloads).
2.  Follow the installation instructions provided by NVIDIA for your operating system.
3.  Ensure `nvcc` is added to your system's PATH environment variable.

### Python Environment Setup
It's recommended to use a virtual environment for Python dependencies.
1.  Create a virtual environment:
    ```bash
    python -m venv .venv 
    ```
    (Or any other name you prefer for the environment folder, e.g., `env`, `mlp_env`)
2.  Activate the virtual environment:
    * On Windows:
        ```bash
        .\.venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
3.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Compilation and Execution

The MLP dimensions (`INPUT_FEATURES_DIM`, `HIDDEN_NEURONS_DIM`, `OUTPUT_NEURONS_DIM` in `MLP_cublas.cu`, and `INPUT_FEATURES`, `HIDDEN_NEURONS`, `OUTPUT_NEURONS` in `MLP_naive.cu`) are compile-time constants. You may modify them directly in the source files and recompile if needed.

Compiled executables (e.g., `mlp_naive_app`, `mlp_cublas_app`) are expected to be generated in the project root directory by default if not using a build subdirectory with CMake.

### 1. Compiling the CUDA Code

#### Using CMake (Recommended)

1.  Create a `CMakeLists.txt` file in the project root. Example content:
    ```cmake
    cmake_minimum_required(VERSION 3.10)
    project(CudaMLP LANGUAGES CXX CUDA)

    find_package(CUDA REQUIRED)

    add_executable(mlp_naive_app MLP_naive.cu)
    set_target_properties(mlp_naive_app PROPERTIES CUDA_ARCHITECTURES "XX") # e.g., "75" or "86"

    add_executable(mlp_cublas_app MLP_cublas.cu)
    set_target_properties(mlp_cublas_app PROPERTIES CUDA_ARCHITECTURES "XX") # e.g., "75" or "86"
    target_link_libraries(mlp_cublas_app PRIVATE ${CUDA_CUBLAS_LIBRARIES})
    ```
    Replace `"XX"` with your GPU's compute capability (e.g., `75`, `86`). Find yours [here](https://developer.nvidia.com/cuda-gpus).

2.  Create a build directory (optional but good practice) and navigate into it:
    ```bash
    mkdir build
    cd build
    cmake ..
    make # or cmake --build .
    ```
    Executables will be in the `build` directory. If you run `cmake .` in the root, they will be in the root.

#### Manual Compilation with `nvcc` (from the project root)

1.  **Compile the naive implementation:**
    ```bash
    nvcc MLP_naive.cu -o mlp_naive_app -arch=sm_XX
    ```
2.  **Compile the cuBLAS-optimized implementation:**
    ```bash
    nvcc MLP_cublas.cu -o mlp_cublas_app -arch=sm_XX -lcublas
    ```
    Replace `sm_XX` with your GPU's compute architecture (e.g., `sm_75`, `sm_86`). Executables will be created in the project root.

### 2. Running the Applications

#### Directly
After compilation, run the applications from their location (e.g., project root or `build/`) providing the required command-line arguments:

**For `mlp_naive_app`:**
```bash
./mlp_naive_app <batch_size> <threads_per_block_x> <threads_per_block_y>
```
Example:
```bash
./mlp_naive_app 1024 16 16
```

**For `mlp_cublas_app`:**
(The `threads_per_block_x` and `threads_per_block_y` arguments are for the custom bias/ReLU kernels.)
```bash
./mlp_cublas_app <batch_size> <threads_per_block_x_for_bias_relu> <threads_per_block_y_for_bias_relu>
```
Example:
```bash
./mlp_cublas_app 1024 16 16
```

Both applications will print a CSV-formatted line to standard output containing the input parameters and the measured GPU execution time in milliseconds. A warning will be printed to `stderr` if the total threads per block (`threads_per_block_x * threads_per_block_y`) is not a multiple of 32.

**Output CSV Format:**
For `mlp_naive_app`:
`BatchSize,TPB_X,TPB_Y,TotalThreadsPerBlock,GPUTime_ms`

For `mlp_cublas_app`:
`BatchSize,TPB_X_BiasKernel,TPB_Y_BiasKernel,TotalThreads_BiasKernel,GPUTime_ms`

#### Using `execute.py` (Assumed Functionality)
The `execute.py` script is likely designed to automate the execution of the compiled CUDA applications, possibly iterating over various `batch_size` and thread configurations, and collecting their CSV outputs into a single file or directly feeding them into an analysis workflow.

### 3. Analyzing Results
The `analysis.ipynb` Jupyter Notebook can be used to load, process, and visualize the performance data (CSV output) generated by the MLP applications.
1.  Ensure you have activated your Python virtual environment.
2.  Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
3.  Open `analysis.ipynb` from the Jupyter interface in your browser.

## Implemented Features

* **MLP Architecture:** Input Layer -> Hidden Layer (with ReLU activation) -> Output Layer (linear).
    * `MLP_naive.cu` network dimensions: `INPUT_FEATURES`, `HIDDEN_NEURONS`, `OUTPUT_NEURONS`.
    * `MLP_cublas.cu` network dimensions: `INPUT_FEATURES_DIM`, `HIDDEN_NEURONS_DIM`, `OUTPUT_NEURONS_DIM`.
* **Naive Implementation (`MLP_naive.cu`):**
    * Custom CUDA kernels (`forward_layer1_naive_kernel`, `forward_layer2_naive_kernel`) for combined matrix multiplication, bias addition, and ReLU activation (layer 1) or linear output (layer 2).
* **cuBLAS Implementation (`MLP_cublas.cu`):**
    * Matrix multiplications performed using `cublasSgemm`.
    * Custom CUDA kernels (`add_bias_relu_kernel`, `add_bias_kernel`) for element-wise bias addition and activation.
* **Performance Measurement:**
    * GPU execution time is measured using CUDA Events.
* **Error Handling:**
    * Robust error checking for CUDA API calls (`cudaSafeCall`) and cuBLAS API calls (`cublasSafeCall`).
* **Data Initialization:**
    * Input data ($X$), weights ($W_1, W_2$), and biases ($b_1, b_2$) are initialized with random floating-point values on the host and then copied to the device.
* **Configurable Parameters:**
    * Batch size and threads per block (for custom kernels) are configurable via command-line arguments for the CUDA applications.
* **Execution Automation & Analysis:**
    * `execute.py` for potentially automating benchmark runs.
    * `analysis.ipynb` for data visualization and comparison.

## Expected Results / Performance Comparison

The `mlp_cublas_app` is expected to demonstrate significantly lower execution times (i.e., higher performance) compared to `mlp_naive_app`, especially as the `BATCH_SIZE` and network dimensions increase. The `analysis.ipynb` notebook should provide visual confirmation of these trends.

## Possible Future Improvements

* Implement MLP training (backpropagation) for both versions.
* Support for configurable network architectures at runtime.
* Explore other activation functions.
* Further optimize custom kernels in `MLP_naive.cu` using techniques like shared memory tiling.
* Compare with other GPU-accelerated libraries like cuDNN.
* Implement loading of datasets and weights from files.
* Enhance `execute.py` for more comprehensive benchmarking (e.g., multiple runs, averaging, warm-up iterations).
