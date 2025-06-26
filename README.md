# High-Performance Computing Projects: CPU & GPU Optimization

This repository contains two projects developed for the Computer Architecture course at the University of Pisa's Master's Degree in Artificial Intelligence and Data Engineering.

The projects explore application optimization on two different hardware architectures: a multi-core CPU and a massively parallel GPU. Both demonstrate a common methodology: profiling a baseline "naive" implementation to identify performance bottlenecks, then engineering an optimized solution to maximize performance and resource utilization.

---

## Projects

### 1. CPU Optimization: High-Performance Text Processing

This project provides a deep dive into optimizing a multi-threaded C++ text processing application. It covers the process of:
* **Profiling:** Identifying that the standard C++ `<regex>` library was a critical bottleneck, consuming over 92% of execution time. 
* **Optimization:** Replacing the standard library with Google's high-performance `re2` engine. 
* **Results:** Achieving a **4x to 8x increase in throughput** and demonstrating excellent scalability on multi-core processors. 

**[>> Go to the full CPU Project README](CA-project-CPU/readme.md)**

### 2. GPU Optimization: Accelerating a Neural Network with CUDA & cuBLAS

This project is a comparative study of two CUDA C++ implementations of a Multi-Layer Perceptron (MLP). It highlights the performance difference between:
* **A "naive" kernel:** A custom-written CUDA kernel for matrix multiplication.
* **An optimized solution:** Using the `cublasSgemm` routine from NVIDIA's highly-optimized cuBLAS library.
* **Results:** The cuBLAS implementation shows an immense **>22x increase in computational throughput (GFLOPS)**, demonstrating the power of vendor-optimized libraries for common deep learning operations.

**[>> Go to the full GPU Project README](CA-project-GPU/README.md)**

---

## 👥 Authors

Both projects were developed by:

* Mattia Gemelli
* Francesco Nocella

## 📜 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**. See the [LICENSE](http://creativecommons.org/licenses/by-nc-sa/4.0/) file for details.

This means the software is provided "as-is" and you are free to use, share, and adapt it for **non-commercial purposes**, as long as you provide appropriate attribution and distribute any derivative works under the same license.
