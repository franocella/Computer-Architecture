import os
import sys
import subprocess
import gdown

def generate_shortened_filename(original_path):
    """
    Generate a shortened filename based on the original.
    For example: "collection.tsv" -> "collection_shortened.tsv"
    """
    base, ext = os.path.splitext(original_path)
    return f"{base}_shortened{ext}"

def download_dataset(url, output_path):
    """
    Download the dataset from the specified URL if it doesn't already exist.
    """
    if not os.path.exists(output_path):
        print(f"[Download] Dataset not found. Downloading dataset to {output_path}...")
        gdown.download(url=url, output=output_path, fuzzy=True, quiet=False)
        print("[Download] Dataset downloaded successfully.")
    else:
        print(f"[Download] Dataset already exists: {output_path}")

def create_short_version(input_file, num_lines, force_update=False):
    """
    Create a shortened version of the input file by reading the first 'num_lines' lines.
    If force_update is True, the shortened file will be overwritten.
    The shortened file is saved with the name generated by generate_shortened_filename.
    """
    output_file = generate_shortened_filename(input_file)
    if os.path.exists(output_file) and not force_update:
        print(f"[Shorten] Shortened file already exists: {output_file}")
        return output_file

    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        file_size = os.path.getsize(input_file)
        print(f"[Shorten] Processing file: {input_file} ({file_size/(1024**2):.2f} MB)")
        print(f"[Shorten] Creating a short version with {num_lines} lines...")

        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:

            # Try to use tqdm for a progress bar if available
            try:
                from tqdm import tqdm
                for _ in tqdm(range(num_lines), desc="Processing lines"):
                    f_out.write(next(f_in))
            except (ImportError, StopIteration):
                if 'tqdm' not in sys.modules:
                    print("[Shorten] Tip: Install tqdm for a progress bar (pip install tqdm)")
                for i in range(num_lines):
                    try:
                        f_out.write(next(f_in))
                        if i % 1000 == 0 and i > 0:
                            print(f"[Shorten] Processed {i}/{num_lines} lines...")
                    except StopIteration:
                        print(f"[Shorten] Warning: File ended at line {i+1}.")
                        break

        print(f"[Shorten] Successfully created shortened file: {output_file}")
        return output_file

    except Exception as e:
        print(f"[Error] {str(e)}")
        sys.exit(1)

def run_stemmer_executable(exec_name, thread_count, results_file):
    """
    Execute the specified executable with the given number of threads.
    The executable should be located in the same folder as this script.
    """
    # Determine the directory where the script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the full path to the executable
    executable_path = os.path.join(current_dir, exec_name)

    # Check if the executable exists in the same folder
    if not os.path.isfile(executable_path):
        print(f"[Executable] Error: The executable '{exec_name}' was not found in the directory {current_dir}")
        return

    print(f"[Executable] Running {exec_name} with {thread_count} threads. Results will be saved in {results_file}.")
    try:
        # Run the executable (blocking call)
        # The number of workers (thread_count) is the first argument
        subprocess.run([executable_path, str(thread_count), results_file])
        print(f"[Executable] Finished running {exec_name} with {thread_count} threads.")
    except Exception as e:
        print(f"[Executable] Exception during execution of {exec_name}: {e}")

def main():
    # URL of the dataset (modify the URL as needed)
    dataset_url = 'https://drive.google.com/file/d/1cSGfMEhFwlKUcfTe250ld-Jkj_gF111K/view'
    original_file = "collection.tsv"

    print("[Main] Starting the dataset experiments...")

    # 1. Download the complete dataset if not already present
    download_dataset(dataset_url, original_file)

    # Define the sequence of sizes for the shortened file
    dataset_sizes = [10000]
    n_iter= 1
    n_workers_max = 19
    n_workers_min = 1
    # Main loop: for each dataset size, update the shortened file and run experiments.
    for num_lines in dataset_sizes:
        print(f"\n=== Updating dataset: {num_lines} lines ===")
        # Update the shortened file by overwriting it
        create_short_version(original_file, num_lines, force_update=True)

        # Loop for the non-optimized and optimized versions
        for threads in range(n_workers_min, n_workers_max):
            for i in range(n_iter):
                print(f"\n[Experiment] Iteration {i+1}/{n_iter} with {threads} threads (dataset: {num_lines} lines) using non-optimized version")
                run_stemmer_executable("bazel-bin\\preprocessing_no_opt.exe", threads, "performance_stats_no_opt.csv")

                print(f"\n[Experiment] Iteration {i+1}/{n_iter} with {threads} threads (dataset: {num_lines} lines) using optimized version")
                run_stemmer_executable("bazel-bin\\preprocessing_opt.exe", threads, "performance_stats_opt.csv")
        print(f"\n--- Completed experiments for dataset with {num_lines} lines ---\n")

    print("[Main] All experiments completed successfully.")
    print("[Main] Shutting down the system in 30 seconds. Press CTRL+C to cancel.")
    os.system("shutdown /s /t 30")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Main] Operation cancelled by user.")
        sys.exit(0)