# High-Performance CPU Text Processing Pipeline

This project, developed for the Computer Architecture course at the University of Pisa, demonstrates the optimization of a multi-threaded C++ text processing pipeline. By identifying and resolving a critical performance bottleneck, the application's throughput was increased by up to 8x.

## Project Overview

The goal was to optimize a text processing application that performs several key natural language tasks. The pipeline is designed to be highly scalable, leveraging multi-core CPU architectures to process large volumes of text efficiently.

The core optimization strategy involved two main implementations:
1.  A **baseline version** using standard C++ libraries.
2.  An **optimized version** that replaces the bottleneck component with a high-performance, specialized library.

## The Processing Pipeline

The application processes text in three main stages:
1.  **Tokenization**: The input text is segmented into individual words or tokens.
2.  **Stopword Removal**: Common words (e.g., "a", "the", "in") that provide little semantic value are filtered out from the list of tokens.
3.  **Stemming**: Tokens are reduced to their root form (or "stem") to normalize them. For example, "processing" and "processed" both become "process". This is handled by the [`stemming::english_stem<>` library](https://github.com/Blake-Madden/OleanderStemmingLibrary.git)

## Bottleneck Analysis and Optimization

### Initial Implementation (`Preprocessing_no_opt.cpp`)
The initial version of the pipeline was built using the standard C++ `<regex>` library for the tokenization step. Profiling the application under load revealed that this was a major performance bottleneck.

- **Problem:** The `std::wsregex_token_iterator` was found to be responsible for over 92% of the total execution time. Standard library regex engines, while versatile, can suffer from inefficiencies and catastrophic backtracking on complex patterns.

### Optimized Implementation (`Preprocessing_opt.cpp`)
To address the bottleneck, the standard regex library was replaced with **Google's RE2 library**.

- **Solution:** RE2 is a faster and safer regular expression engine designed for high performance. It guarantees that its execution time is linear in the size of the input, avoiding the performance cliffs of other engines.
- **Further Enhancements:** The optimized code also merges the tokenization, stopword removal, and stemming steps into a more cohesive function (`tokenize_filter_and_stem`) and utilizes `thread_local` objects for the RE2 compiler and stemmer, reducing object creation overhead in a multi-threaded context.

## Performance Results

The switch to the RE2 library resulted in a dramatic performance improvement.

- **Throughput:** The optimized version achieved a **4x to 8x increase in throughput** (measured in tokens per second) compared to the baseline.
- **Execution Time:** Execution time was significantly reduced across all thread counts, as shown in the performance comparison charts from the project analysis.
- **Scalability:** The optimized application demonstrates excellent scalability with an increasing number of threads, making full use of available CPU cores.

## How to Run

The project includes an orchestrator script that automates the entire process of downloading the dataset, building the executables, and running the performance tests.

**Dependencies:**
* A C++17 compliant compiler
* Bazel build system
* Python 3
* Google's `re2` library
* Python libraries: `gdown`, `tqdm`

**Instructions:**

1.  **Setup:** Ensure all dependencies are installed and available in your environment.
2.  **Build:** Compile the C++ source files using Bazel.
    ```bash
    # (Example build command, adjust as needed)
    bazel build //...
    ```
    The `orchestrator.py` script expects the compiled executables (`preprocessing_no_opt.exe` and `preprocessing_opt.exe`) to be in the `bazel-bin` directory.

3.  **Execute:** Run the orchestrator script from the project's root directory.
    ```python
    python orchestrator.py
    ```
    The script will first download the required dataset. It will then execute both the non-optimized and optimized versions with a varying number of threads. Performance results will be logged to `performance_stats_no_opt.csv` and `performance_stats_opt.csv`.

## Authors

* M. Gemelli
* F. Nocella