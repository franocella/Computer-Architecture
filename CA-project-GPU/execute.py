import subprocess
import csv
import os

# --- Costanti dell'architettura MLP (devono corrispondere a quelle nei file .cu) ---
# Queste sono necessarie per calcolare i FLOPs.
# Aggiorna questi valori perché corrispondano ai #define nei file .cu che stai testando.
# Esempio basato sui valori forniti in precedenza:
# Per MLP_naive.cu (originale): INPUT_FEATURES = 128, HIDDEN_NEURONS = 1024
# Per MLP_cublas.cu (esempio): INPUT_FEATURES = 512, HIDDEN_NEURONS = 2048
# *** IMPOSTA I VALORI CORRETTI QUI PER L'ESECUTABILE CHE STAI TESTANDO ***
INPUT_FEATURES = 512  # Esempio, cambia se necessario
HIDDEN_NEURONS = 2048 # Esempio, cambia se necessario
OUTPUT_NEURONS = 100
# ---------------------------------------------------------------------------

# Configurazioni da testare
batch_sizes_to_test = [256, 512, 1024, 2048, 4096]

# Configurazioni di Thread per Blocco da testare
# Queste saranno usate per:
# - (tpb_x, tpb_y) in MLP_naive.cu
# - (threads_x_bias, threads_y_bias) in MLP_cublas.cu
thread_block_configs_to_test = [
    # Multipli di 32 per il totale dei thread sono generalmente buoni
    (8, 4),    # 32 threads
    (4, 8),    # 32 threads
    (16, 2),   # 32 threads
    (2, 16),   # 32 threads

    (8, 8),    # 64 threads
    (16, 4),   # 64 threads
    (4, 16),   # 64 threads
    (32, 2),   # 64 threads
    (2, 32),   # 64 threads

    (16, 8),   # 128 threads
    (8, 16),   # 128 threads
    (32, 4),   # 128 threads
    (4, 32),   # 128 threads
    (64, 2),   # 128 threads
    (2, 64),   # 128 threads

    (16, 16),  # 256 threads
    (32, 8),   # 256 threads
    (8, 32),   # 256 threads
    (64, 4),   # 256 threads
    (4, 64),   # 256 threads
    (128, 2),  # 256 threads
    (2, 128),  # 256 threads

    (32, 16),  # 512 threads
    (16, 32),  # 512 threads
    (64, 8),   # 512 threads
    (8, 64),   # 512 threads
    (256, 2),  # 512 threads
    (2, 256),  # 512 threads
    (128, 4),  # 512 threads
    (4, 128),  # 512 threads
    
    (32, 32),  # 1024 threads
    (64, 16),  # 1024 threads
    (16, 64),  # 1024 threads
    (128, 8),  # 1024 threads
    (8, 128),  # 1024 threads
    (256, 4),  # 1024 threads
    (4, 256),  # 1024 threads
]

# --- CONFIGURAZIONE PER QUALE ESEGUIBILE TESTARE ---
# Imposta execution_mode a "naive" per MLP_naive.cu
# Imposta execution_mode a "cublas" per MLP_cublas.cu

execution_mode = "cublas"  # CAMBIA QUI: "naive", "constant" o "cublas"
# ----------------------------------------------------

if execution_mode == "naive":
    cuda_executable_name = "MLP_naive" # Assicurati che questo sia il nome del tuo eseguibile naive
    output_csv_file_template = "mlp_naive_performance_py.csv"
elif execution_mode == "constant":
    cuda_executable_name = "MLP_constant_memory" # Assicurati che corrisponda al tuo eseguibile compilato con memoria costante
    output_csv_file_template = "mlp_constant_performance_py.csv"
elif execution_mode == "cublas":
    cuda_executable_name = "MLP_cublas" # Assicurati che corrisponda al tuo eseguibile compilato con cuBLAS
    output_csv_file_template = "mlp_cublas_performance_py.csv"
else:
    print(f"Errore: Valore di 'execution_mode' ('{execution_mode}') non riconosciuto. Scegli 'naive', 'constant' o 'cublas'.")
    exit(1)


# Determina il nome completo dell'eseguibile e del file di output
if os.name == 'nt': # Windows
    cuda_executable = f"./{cuda_executable_name}.exe"
else: # Linux/macOS
    cuda_executable = f"./{cuda_executable_name}"

output_csv_file = output_csv_file_template

if not os.path.exists(cuda_executable):
    print(f"Errore: Eseguibile CUDA '{cuda_executable}' non trovato.")
    compile_command = f"nvcc {cuda_executable_name}.cu -o {cuda_executable_name} -arch=sm_XX"
    if execution_mode == "cublas":
        compile_command += " -lcublas"
    print(f"Assicurati di averlo compilato (es. {compile_command})")
    exit(1)

def calculate_total_flops(batch_size, input_features, hidden_neurons, output_neurons):
    """
    Calcola il numero approssimativo di FLOPs per un forward pass dell'MLP a 2 strati.
    Layer 1: H = ReLU(X * W1 + b1) (W1 è I x H per convenzione X*W)
    Layer 2: Y = H * W2 + b2 (W2 è H x O per convenzione H*W)
    """
    # Layer 1: X (B, I) * W1 (I, H) -> MatMul (B, H)
    flops_layer1_matmul = batch_size * hidden_neurons * (2 * input_features)
    flops_layer1_bias = batch_size * hidden_neurons # Aggiunta bias
    flops_layer1_relu = batch_size * hidden_neurons # Attivazione ReLU
    flops_layer1 = flops_layer1_matmul + flops_layer1_bias + flops_layer1_relu

    # Layer 2: H (B, H_prev) * W2 (H_prev, O) -> MatMul (B, O)
    flops_layer2_matmul = batch_size * output_neurons * (2 * hidden_neurons) # H_prev è hidden_neurons
    flops_layer2_bias = batch_size * output_neurons # Aggiunta bias
    # Nessuna ReLU sull'output negli esempi C++ forniti per il layer2
    flops_layer2 = flops_layer2_matmul + flops_layer2_bias
    
    total_flops = flops_layer1 + flops_layer2
    return total_flops

with open(output_csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # L'header è lo stesso per entrambi, interpretando TPB_X/Y come parametri per i kernel
    header = ["BatchSize", "TPB_X", "TPB_Y", "TotalThreads_KernelOrAux", "GPUTime_ms", "Samples_per_sec", "GFLOPS"]
    csv_writer.writerow(header)
    print(f"Avvio esperimenti per {cuda_executable_name}... Risultati salvati in {output_csv_file}")
    print(",".join(header)) # Stampa l'header anche sulla console

    for bs in batch_sizes_to_test:
        for tpb_x, tpb_y in thread_block_configs_to_test:
            total_threads_per_block = tpb_x * tpb_y
            
            if total_threads_per_block == 0:
                continue
            if total_threads_per_block > 1024: # Limite comune per GPU
                print(f"Skipping TPB ({tpb_x},{tpb_y}) for BatchSize {bs} - Total threads {total_threads_per_block} exceeds 1024.")
                continue

            print(f"Esecuzione con BatchSize: {bs}, TPB_X: {tpb_x}, TPB_Y: {tpb_y} (Total for kernels/aux: {total_threads_per_block})")
            
            # Il comando è lo stesso per MLP_naive e MLP_cublas (entrambi prendono 3 argomenti dopo il nome)
            command = [cuda_executable, str(bs), str(tpb_x), str(tpb_y)]
            
            try:
                # Timeout aumentato a 5 minuti per test più lunghi
                result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
                
                output_lines = result.stdout.strip().split('\n')
                csv_output_line = ""

                if output_lines:
                    # La riga CSV è attesa come ultima riga informativa
                    csv_output_line = output_lines[-1]
                
                if csv_output_line:
                    print(f"  Output CUDA (linea CSV candidata): {csv_output_line}")
                    parsed_values_str = csv_output_line.split(',')
                    
                    # Atteso: BatchSize, TPB_X, TPB_Y, TotalThreads, GPUTime_ms
                    if len(parsed_values_str) == 5:
                        gpu_time_ms_str = parsed_values_str[4]
                        try:
                            gpu_time_ms = float(gpu_time_ms_str)
                            gpu_time_s = gpu_time_ms / 1000.0

                            samples_per_sec = 0
                            gflops = 0
                            if gpu_time_s > 0: # Evita divisione per zero
                                samples_per_sec = bs / gpu_time_s
                                total_mlp_flops = calculate_total_flops(bs, INPUT_FEATURES, HIDDEN_NEURONS, OUTPUT_NEURONS)
                                gflops = (total_mlp_flops / gpu_time_s) / 1.0e9
                            
                            csv_row = [
                                parsed_values_str[0], 
                                parsed_values_str[1], 
                                parsed_values_str[2], 
                                parsed_values_str[3], 
                                f"{gpu_time_ms:.4f}", 
                                f"{samples_per_sec:.2f}",
                                f"{gflops:.2f}"
                            ]
                            csv_writer.writerow(csv_row)
                            print(f"    -> CSV: {','.join(csv_row)}")

                        except ValueError:
                            print(f"  Errore: Impossibile convertire il tempo GPU '{gpu_time_ms_str}' in float dalla linea: {csv_output_line}")
                    else:
                        # Potrebbe esserci output informativo prima della riga CSV, proviamo la penultima
                        if len(output_lines) > 1 and len(output_lines[-2].split(',')) == 5:
                            print(f"  Info: La linea CSV candidata precedente non era valida. Provo la penultima linea.")
                            csv_output_line = output_lines[-2]
                            parsed_values_str = csv_output_line.split(',')
                            gpu_time_ms_str = parsed_values_str[4]
                            # Ripeti il blocco try-except per il parsing
                            try:
                                gpu_time_ms = float(gpu_time_ms_str)
                                gpu_time_s = gpu_time_ms / 1000.0
                                samples_per_sec = 0
                                gflops = 0
                                if gpu_time_s > 0:
                                    samples_per_sec = bs / gpu_time_s
                                    total_mlp_flops = calculate_total_flops(bs, INPUT_FEATURES, HIDDEN_NEURONS, OUTPUT_NEURONS)
                                    gflops = (total_mlp_flops / gpu_time_s) / 1.0e9
                                csv_row = [
                                    parsed_values_str[0], parsed_values_str[1], parsed_values_str[2], parsed_values_str[3],
                                    f"{gpu_time_ms:.4f}", f"{samples_per_sec:.2f}", f"{gflops:.2f}"
                                ]
                                csv_writer.writerow(csv_row)
                                print(f"    -> CSV (da penultima linea): {','.join(csv_row)}")
                            except ValueError:
                                print(f"  Errore: Impossibile convertire il tempo GPU '{gpu_time_ms_str}' in float dalla penultima linea: {csv_output_line}")
                        else:
                            print(f"  Errore: La linea CSV candidata '{csv_output_line}' non ha 5 campi come atteso (ne ha {len(parsed_values_str)}).")
                else: 
                    stdout_content = result.stdout.strip() if result.stdout else "Nessun output stdout."
                    print(f"  Attenzione: Nessuna linea CSV valida trovata nell'output stdout per BatchSize {bs}, TPB ({tpb_x},{tpb_y}). Output stdout completo:\n{stdout_content}")

                stderr_output = result.stderr.strip()
                if stderr_output:
                    if "Warning:" in stderr_output or \
                       "not a multiple of 32" in stderr_output or \
                       "shared memory requested" in stderr_output or \
                       "potrebbe essere eccessiva per SM" in stderr_output :
                        print(f"  Avviso STDERR dal programma CUDA:\n{stderr_output}")
                    else: 
                        print(f"  Errore STDERR dal programma CUDA:\n{stderr_output}")

            except subprocess.CalledProcessError as e:
                print(f"  Errore durante l'esecuzione di {cuda_executable} con BatchSize {bs}, TPB ({tpb_x},{tpb_y}):")
                print(f"  Return code: {e.returncode}")
                print(f"  Stdout: {e.stdout.strip() if e.stdout else 'N/A'}")
                print(f"  Stderr: {e.stderr.strip() if e.stderr else 'N/A'}")
            except subprocess.TimeoutExpired:
                print(f"  Timeout durante l'esecuzione di {cuda_executable} con BatchSize {bs}, TPB ({tpb_x},{tpb_y}).")
            except Exception as e:
                print(f"  Si è verificato un errore Python imprevisto: {e}")

print(f"Esperimenti completati. Risultati in {output_csv_file}")