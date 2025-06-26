// Preprocessing_opt.cpp

// System headers
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h> // For sched_setaffinity if needed on Linux (though not used here)
#endif

// Standard library headers
#include <algorithm>
#include <atomic>
#include <chrono>
#include <codecvt>
#include <ctime>
#include <fstream>
#include <iostream>
#include <locale>
#include <mutex>
// #include <regex> // No longer needed if only using RE2
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <limits>
#include <utility> // For std::move
#include <numeric> // For std::accumulate (optional alternative for sum)

// Third-party library headers
#include "stemmer/english_stem.h" // Include your stemmer library header
#include <re2/re2.h>              // Include RE2 header

// Constants
constexpr const char *LOG_FILE = "processing_log.csv";
constexpr const char *DATA_FILE = "processed_data.csv";
constexpr const char *STATS_FILENAME = "performance.csv";
constexpr int NUMTHREADS = 12;     // Default number of threads
constexpr int MAXWORKERS = 26;     // Maximum allowed workers

// Regex pattern for tokenization (alphanumeric sequences with optional internal hyphens/underscores)
const std::string pattern_str = R"(([[:alnum:]]+([-_][[:alnum:]]+)*))";

// Forward declarations
struct Document;
std::wstring to_wstring(const std::string &str);
std::string to_string(const std::wstring &wstr);
std::vector<std::wstring> tokenize_filter_and_stem(const std::wstring &text, const RE2 &re, stemming::english_stem<> &stemmer, unsigned long &count_accumulator);
std::vector<std::wstring> processDocument(Document &doc, const RE2 &re, stemming::english_stem<> &stemmer, unsigned long &thread_local_token_count);
void threadTask(int core_id, unsigned int num_threads, unsigned long &thread_token_accumulator);


// Data structures
struct Document
{
    std::wstring content;             // Original content (cleared after processing)
    std::vector<std::wstring> tokens; // Processed tokens (stemmed, no stopwords)
    int id;                           // Document identifier
};


// Global state (consider better encapsulation in a larger system)
std::vector<Document> documents;      // Global vector holding all documents
std::mutex log_mutex;                 // Mutex to protect log file access
std::atomic<int> lastUpdatedIndex(0); // Atomic index for work distribution among threads
// Note: Global atomic 'number_of_tokens' is removed, using thread-local counts instead.


// Stopwords set (must be lowercase for consistency with the processing)
const std::unordered_set<std::wstring> stopwords = {
    L"a", L"an", L"and", L"are", L"as", L"at", L"be", L"but", L"by",
    L"for", L"if", L"in", L"into", L"is", L"it", L"no", L"not", L"of",
    L"on", L"or", L"such", L"that", L"the", L"their", L"then", L"there",
    L"these", L"they", L"this", L"to", L"was", L"will", L"with"};

// Utility functions

// Gets the current time as a formatted string
std::string get_current_time()
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm tm_struct;
#if defined(_WIN32)
    localtime_s(&tm_struct, &now_time);
#else
    localtime_r(&now_time, &tm_struct);
#endif
    char time_buf[80];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &tm_struct);
    return std::string(time_buf);
}

// Converts a UTF-8 std::string to std::wstring
std::wstring to_wstring(const std::string &str)
{
#ifdef _WIN32
    if (str.empty()) return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size_needed);
    return wstr;
#else
    // Using codecvt is deprecated in C++17, but standard alternatives are complex.
    // Consider using a library like ICU if maximum portability/correctness is needed.
    // This implementation might work on many Linux systems.
    try {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        return converter.from_bytes(str);
    } catch (const std::range_error& e) {
        log_message("Error converting string to wstring: " + std::string(e.what()) + " - String: " + str, true);
        return L""; // Return empty on error
    }
#endif
}

// Converts a std::wstring to UTF-8 std::string
std::string to_string(const std::wstring &wstr)
{
#ifdef _WIN32
    if (wstr.empty()) return "";
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string str(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &str[0], size_needed, NULL, NULL);
    return str;
#else
    // Using codecvt is deprecated in C++17. See note in to_wstring.
    try {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        return converter.to_bytes(wstr);
    } catch (const std::range_error& e) {
         log_message("Error converting wstring to string: " + std::string(e.what()), true);
         return ""; // Return empty on error
    }
#endif
}

// Core functionality



// Logs a message to the console and optionally to a log file (thread-safe)
void log_message(const std::string &message, bool error = false)
{
    std::unique_lock<std::mutex> lock(log_mutex); // Lock mutex for thread safety
    std::ofstream log_file(LOG_FILE, std::ios_base::app | std::ios_base::out);

    std::stringstream ss;
    ss << "[" << get_current_time() << "] "
       << "[Thread " << std::this_thread::get_id() << "] "
       << (error ? "[ERROR] " : "[INFO] ") << message;
    const std::string final_msg = ss.str();

    if (log_file.is_open())
    {
        log_file << final_msg << "\n";
        log_file.flush(); // Ensure message is written immediately
    }
    else
    {
        // Log failure to stderr as well
        std::cerr << "LOG ERROR: Failed to open log file '" << LOG_FILE << "'\n";
    }

    // Also print to console
    if (error)
        std::cerr << final_msg << std::endl;
    else
        std::cout << final_msg << std::endl;
}

// Tokenizes, filters stopwords, and stems the input text using pre-compiled RE2 and stemmer objects.
// Updates the provided count_accumulator with the number of tokens added.
std::vector<std::wstring> tokenize_filter_and_stem(const std::wstring &text,
                                                   const RE2 &re,                      // Pass compiled RE2 object by const reference
                                                   stemming::english_stem<> &stemmer, // Pass stemmer object by reference
                                                   unsigned long &count_accumulator)  // Pass accumulator by reference
{
    std::vector<std::wstring> processed_tokens;
    // Optional: Reserve memory to potentially avoid reallocations
    // Adjust the division factor based on expected average token length
    size_t estimated_tokens = text.length() / 5;
    processed_tokens.reserve(estimated_tokens > 10 ? estimated_tokens : 10); // Reserve some space

    std::string utf8_text = to_string(text); // Convert to UTF-8 for RE2
    re2::StringPiece input(utf8_text);
    std::string token_str;                   // String to hold the matched token from RE2
    unsigned long local_token_count = 0;     // Count tokens processed in this call

    // Use RE2's FindAndConsume for efficient iteration
    while (RE2::FindAndConsume(&input, re, &token_str))
    {
        if (!token_str.empty())
        {
            std::wstring wtoken = to_wstring(token_str); // Convert token to wstring for stopword/stemming

            local_token_count++;

            if (stopwords.find(wtoken) == stopwords.end())
            {
                // Not a stopword
                // 2. Stemming (in-place using the passed stemmer object)
                stemmer(wtoken);

                // 3. Add stemmed token to result vector
                processed_tokens.push_back(std::move(wtoken)); // Use move for potential efficiency
            }
        }
    }

    // Update the thread-local accumulator passed by reference
    count_accumulator += local_token_count;

    return processed_tokens;
}

// Processes a single document: lowercases, tokenizes, filters, stems.
// Uses thread-local RE2 and stemmer objects passed down from threadTask.
std::vector<std::wstring> processDocument(Document &doc,
                                          const RE2 &re,                      // Pass compiled RE2 object
                                          stemming::english_stem<> &stemmer, // Pass stemmer object
                                          unsigned long &thread_token_accumulator) // Pass accumulator
{
    // Create a mutable copy of the content
    std::wstring content_copy = doc.content;

    // Lowercase the entire content copy once.
    // This is needed for the case-sensitive stopword check, even if RE2 matching is case-insensitive.
    std::transform(content_copy.begin(), content_copy.end(), content_copy.begin(),
                   [](wchar_t c){ return std::towlower(c); });

    // Call the combined processing function, passing down the thread-local objects and accumulator
    std::vector<std::wstring> tokens = tokenize_filter_and_stem(content_copy, re, stemmer, thread_token_accumulator);

    return tokens;
}

// Loads a TSV dataset (assuming format: doc_id\tcontent) from a file
std::vector<Document> load_tsv_dataset(const std::string &filename)
{
    std::vector<Document> loaded_docs;
    std::ifstream file(filename, std::ios::binary); // Open in binary mode to handle encodings better

    if (!file.is_open())
    {
        // Use log_message for consistency, although this might happen before threads start
        log_message("Failed to open dataset file: " + filename, true);
        throw std::runtime_error("Failed to open dataset file: " + filename);
    }

    // Handle potential UTF-8 BOM (Byte Order Mark)
    constexpr size_t BOM_SIZE = 3;
    char bom[BOM_SIZE];
    if (file.read(bom, BOM_SIZE) &&
        static_cast<unsigned char>(bom[0]) == 0xEF &&
        static_cast<unsigned char>(bom[1]) == 0xBB &&
        static_cast<unsigned char>(bom[2]) == 0xBF)
    {
        // BOM detected and skipped
         log_message("UTF-8 BOM detected and skipped in " + filename, false);
    }
    else
    {
        // No BOM found, rewind to the beginning
        file.clear(); // Clear potential error flags
        file.seekg(0);
    }

    std::string line;
    size_t current_id = 0; // Use size_t for potentially large IDs
    size_t line_num = 0;
    while (std::getline(file, line))
    {
        line_num++;
        // Handle potential Windows line endings (\r\n)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        const size_t tab_pos = line.find('\t');

        // Basic validation: ensure tab exists and there's content after it
        if (tab_pos == std::string::npos /*|| tab_pos >= line.size() - 1*/) // Allow empty content after tab? Decide based on data.
        {                                                                  // Assuming empty content is invalid for now.
             if (tab_pos == std::string::npos) {
                  log_message("Skipping malformed line #" + std::to_string(line_num) + " (no tab found)", false);
             } else {
                  log_message("Skipping malformed line #" + std::to_string(line_num) + " (empty content after tab)", false);
             }
             continue;
        }

        Document doc;
        try
        {
            // Assuming the part before the tab is an ID we don't need to parse currently
            std::string content_str = line.substr(tab_pos + 1);
            // if (content_str.empty()) // Check again if needed based on decision above
            // {
            //     throw std::runtime_error("No content after tab");
            // }
            doc.content = to_wstring(content_str); // Convert content to wstring
            if (doc.content.empty() && !content_str.empty()) {
                 // Log conversion error if original wasn't empty but result is
                 log_message("Warning: Content conversion resulted in empty wstring for line #" + std::to_string(line_num), true);
            }
        }
        catch (const std::exception &e) // Catch potential errors during substring or conversion
        {
            log_message("Error processing line #" + std::to_string(line_num) + ": " + e.what(), true);
            continue; // Skip this document
        }
        doc.id = static_cast<int>(current_id++); // Assign sequential ID
        loaded_docs.push_back(std::move(doc));     // Move document into the vector
    }
    log_message("Finished loading dataset. " + std::to_string(loaded_docs.size()) + " documents loaded.", false);
    return loaded_docs;
}

// Task executed by each worker thread.
// Processes documents from the global 'documents' vector until none are left.
// Uses thread-local RE2 and stemmer objects for efficiency.
// Accumulates token count in a variable passed by reference from main.

void threadTask(int core_id, unsigned int num_threads, unsigned long &thread_token_accumulator)
{
    // --- Thread-Local Objects Initialization ---
    // These objects are created once per thread and reused across multiple document processings.

    // RE2 Options (initialized once per thread)
    thread_local RE2::Options options;
    thread_local bool options_initialized = false;
    if (!options_initialized) {
        options.set_case_sensitive(false); // Ignore case for matching
        options.set_encoding(RE2::Options::EncodingUTF8);
        options.set_posix_syntax(true);    // Use POSIX syntax
        options_initialized = true;
    }

    // Compiled RE2 object (created once per thread)
    // Uses the global 'pattern_str'
    thread_local RE2 re(pattern_str, options);
    if (!re.ok()) {
        // Log error and potentially exit thread if regex compilation fails
        log_message("FATAL: RE2 regex compilation failed in thread: " + re.error(), true);
        return; // Exit this thread
    }

    // Stemmer object (created once per thread)
    thread_local stemming::english_stem<> stemmer;

    // --- End Thread-Local Objects Initialization ---

    // Initialize this thread's token counter (passed by reference from main)
    thread_token_accumulator = 0;

    // Main processing loop for this thread
    while (true)
    {
        // Atomically fetch and increment the index of the next document to process
        int index = lastUpdatedIndex.fetch_add(1, std::memory_order_relaxed);

        // If the index is out of bounds, all documents are being processed or done
        if (index >= static_cast<int>(documents.size())) {
            break; // Exit the loop, thread has no more work
        }

        // Get a reference to the document
        Document &doc = documents[index];

        // Process the document using the thread-local objects and update the accumulator
        try {
             doc.tokens = processDocument(doc, re, stemmer, thread_token_accumulator);
        } catch (const std::exception& e) {
             log_message("Exception during processing document ID " + std::to_string(doc.id) + ": " + e.what(), true);
             // Decide how to handle: continue? clear tokens? rethrow? For now, just log.
             doc.tokens.clear(); // Clear tokens to indicate failure
        }


        // Free memory of the original content string as it's no longer needed
        doc.content.clear();
        doc.content.shrink_to_fit(); // Request the capacity to be reduced
    }
}

// Saves the processed tokens to a CSV file (doc_id, "token1,token2,...")
void save_results_to_csv(const std::vector<std::vector<std::wstring>> &results,
                         const std::string &filename)
{
    log_message("Saving results to " + filename + "...", false);
    try
    {
        std::ofstream csv_file(filename);
        if (!csv_file) {
            throw std::runtime_error("Failed to create CSV file: " + filename);
        }

        // Write CSV Header
        csv_file << "doc_id,tokens\n";

        // Write data rows
        for (size_t i = 0; i < results.size(); ++i)
        {
            csv_file << i << ",\""; // Using doc index as ID, consider using doc.id if available/needed
            std::string token_line;
            bool first_token = true;
            for (const auto &token : results[i])
            {
                if (!first_token) {
                    token_line += ",";
                }
                // Handle potential quotes within tokens if necessary (basic CSV escaping)
                std::string token_str = to_string(token);
                size_t pos = token_str.find('"');
                while (pos != std::string::npos) {
                    token_str.replace(pos, 1, "\"\""); // Escape quotes by doubling them
                    pos = token_str.find('"', pos + 2);
                }
                token_line += token_str;
                first_token = false;
            }
            csv_file << token_line << "\"\n";
        }
        log_message("Successfully saved " + std::to_string(results.size()) + " records to " + filename, false);
    }
    catch (const std::exception &e)
    {
        log_message("CSV Saving Error: " + std::string(e.what()), true);
        // Rethrow or handle as appropriate
        throw;
    }
}

// Sets thread affinity to a specific core ID (Platform-dependent)
void set_thread_affinity(int core_id)
{
#if defined(_WIN32)
    DWORD_PTR mask = 1ULL << core_id;
    if (!SetThreadAffinityMask(GetCurrentThread(), mask))
    {
        // Use GetLastError() for more specific error information on Windows
        DWORD error_code = GetLastError();
        log_message("Warning: Failed to set affinity for core " + std::to_string(core_id) + ". Error code: " + std::to_string(error_code), true);
    }
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        log_message("Warning: Failed to set affinity for core " + std::to_string(core_id) + ". Error code: " + std::to_string(rc), true);
    }
#else
    // Affinity setting not implemented for this platform (e.g., macOS)
    // Log this only once or periodically to avoid spamming logs
    // log_message("Thread affinity setting is not supported/implemented on this platform.", false);
#endif
}

// Saves performance statistics to a CSV file
void save_performance_stats(const std::string &filename, unsigned long num_tokens, size_t num_docs, int num_workers, std::chrono::duration<double> total_time)
{
    log_message("Saving performance stats to " + filename + "...", false);
    try
    {
        // Open in append mode
        std::ofstream stats_file(filename, std::ios::app);
        if (!stats_file.is_open())
        {
            throw std::runtime_error("Failed to open/create performance stats file: " + filename);
        }

        // Check if the file is empty or newly created to write the header
        stats_file.seekp(0, std::ios::end); // Go to the end
        if (stats_file.tellp() == 0) // If the end is at position 0, the file is empty
        {
            stats_file << "num_workers,num_tokens,num_docs,total_time_sec\n";
        }

        // Append the current stats
        stats_file << num_workers << ","
                   << num_tokens << ","
                   << num_docs << ","
                   << total_time.count() << "\n";

        stats_file.flush(); // Ensure data is written
        log_message("Successfully saved performance stats.", false);
    }
    catch (const std::exception &e)
    {
        log_message("Performance Stats Saving Error: " + std::string(e.what()), true);
        // Decide if this error is critical
    }
}

// Main function
int main(int argc, char *argv[])
{
    // --- Argument Parsing ---
    unsigned int numThreads = NUMTHREADS; // Default thread count
    std::string stats_name = STATS_FILENAME; // Default stats filename

    // if (argc < 2)
    // {
    //     // No arguments provided, use defaults (consider this debug/dev mode)
    //     log_message("DEBUG mode active: Using default thread count (" + std::to_string(NUMTHREADS) +
    //                 ") and default stats filename (" + STATS_FILENAME + ")", false);
    // }
    // else
    // {
    //     // Parse thread count from argv[1]
    //     try
    //     {
    //         int requestedThreads = std::stoi(argv[1]);
    //         if (requestedThreads < 1) {
    //              log_message("Warning: Requested thread count " + std::to_string(requestedThreads) + " is less than 1. Using 1 thread.", false);
    //              numThreads = 1u;
    //         } else if (requestedThreads > MAXWORKERS) {
    //              log_message("Warning: Requested thread count " + std::to_string(requestedThreads) + " exceeds maximum " + std::to_string(MAXWORKERS) + ". Using " + std::to_string(MAXWORKERS) + " threads.", false);
    //              numThreads = static_cast<unsigned int>(MAXWORKERS);
    //         } else {
    //              numThreads = static_cast<unsigned int>(requestedThreads);
    //         }
    //     }
    //     catch (const std::invalid_argument& ia) {
    //         std::cerr << "Error: Invalid argument for thread count: " << argv[1] << ". Using default: " << NUMTHREADS << "\n";
    //          numThreads = NUMTHREADS; // Use default on error
    //     }
    //     catch (const std::out_of_range& oor) {
    //          std::cerr << "Error: Thread count argument out of range: " << argv[1] << ". Using default: " << NUMTHREADS << "\n";
    //           numThreads = NUMTHREADS; // Use default on error
    //     }


    //     // Parse optional stats filename from argv[2]
    //     if (argc >= 3)
    //     {
    //         stats_name = argv[2];
    //         // Basic validation: ensure it ends with .csv
    //         size_t pos = stats_name.find_last_of('.');
    //         if (pos == std::string::npos || stats_name.substr(pos) != ".csv")
    //         {
    //              log_message("Warning: Stats filename '" + stats_name + "' does not end with .csv. Appending '.csv'.", false);
    //              stats_name += ".csv";
    //         }
    //     }
    // }
    // --- End Argument Parsing ---


    // Initialize log file (truncate existing content)
    std::ofstream log_init(LOG_FILE, std::ios::trunc);
    if (!log_init) {
         std::cerr << "CRITICAL ERROR: Failed to open or truncate log file: " << LOG_FILE << std::endl;
         return 1; // Cannot proceed without logging
    }
    log_init.close();
    log_message("Starting document processing pipeline with " + std::to_string(numThreads) + " threads.", false);


    try
    {
        // --- Load Dataset ---
        log_message("Loading dataset...", false);
        // Ensure the dataset file exists and is accessible
        // The filename could be made configurable via command line argument too
        documents = load_tsv_dataset("collection_shortened.tsv");
        log_message("Loaded " + std::to_string(documents.size()) + " documents.", false);
        if (documents.empty()) {
             log_message("Warning: Dataset is empty. Nothing to process.", true);
             return 0; // Exit cleanly if no documents
        }

        // --- Thread Pool Setup ---
        // Determine the number of available hardware concurrency units (cores/hyperthreads)
        unsigned int numCores = std::thread::hardware_concurrency();
        if (numCores == 0)
        {
            numCores = 1; // Fallback if detection fails
            log_message("Warning: Could not detect hardware concurrency. Using single core assumption.", true);
        }
        log_message("System reports " + std::to_string(numCores) + " hardware concurrency units.", false);
        if (numThreads > numCores) {
             log_message("Warning: Using more threads (" + std::to_string(numThreads) + ") than available cores (" + std::to_string(numCores) + "). This might lead to context switching overhead.", false);
        }


        log_message("Initializing " + std::to_string(numThreads) + " worker threads...", false);
        std::vector<std::thread> threads;
        threads.reserve(numThreads);
        // Vector to store token counts from each thread
        std::vector<unsigned long> thread_token_counts(numThreads, 0);


        // --- Start Timer and Launch Threads ---
#ifdef _WIN32
        // Define SetThreadDescription function pointer type (for Windows 10+)
        typedef HRESULT(WINAPI * SetThreadDescriptionType)(HANDLE, PCWSTR);
#endif
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();

        for (unsigned int i = 0; i < numThreads; ++i)
        {
            const int target_core = i % numCores; // Assign cores round-robin
            // Capture necessary variables for the thread lambda
            // Capture thread_token_counts by reference (&) to allow modification
            // Capture loop index i by value to index the correct counter
            threads.emplace_back([target_core, numThreads, i, &thread_token_counts] // Correct capture list
            {
                // --- Inside the thread ---
                set_thread_affinity(target_core); // Attempt to set affinity

                // Set thread name (for debugging/profiling) - Platform specific
#ifdef __linux__
                std::string thread_name_str = "Worker_" + std::to_string(target_core);
                pthread_setname_np(pthread_self(), thread_name_str.c_str());
#elif defined(_WIN32)
                HMODULE kernel32 = GetModuleHandleW(L"kernel32");
                if (kernel32) {
                    auto SetThreadDescription = (SetThreadDescriptionType)GetProcAddress(kernel32, "SetThreadDescription");
                    if (SetThreadDescription) {
                        std::wstring thread_name_wstr = L"Worker_" + std::to_wstring(target_core);
                        SetThreadDescription(GetCurrentThread(), thread_name_wstr.c_str());
                    }
                }
#elif defined(__APPLE__) // macOS using pthread_setname_np (different signature)
                 std::string thread_name_str = "Worker_" + std::to_string(target_core);
                 pthread_setname_np(thread_name_str.c_str());
#endif
                // Call the thread task function, passing the reference to this thread's counter element
                threadTask(target_core, numThreads, thread_token_counts[i]);
            });
        }
        log_message("All worker threads launched.", false);


        // --- Wait for Threads to Complete ---
        log_message("Waiting for worker threads to join...", false);
        for (auto &t : threads)
        {
            if (t.joinable())
            {
                try { t.join(); }
                catch (const std::system_error &e) { // Catch potential exceptions from join
                     log_message("Error joining thread: " + std::string(e.what()) + " Code: " + std::to_string(e.code().value()), true);
                 }
                 catch (const std::exception &e) {
                     log_message("Generic error joining thread: " + std::string(e.what()), true);
                 }
            }
        }
        log_message("All worker threads joined.", false);


        // --- Stop Timer ---
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = end_time - start_time;
        log_message("Total processing time: " + std::to_string(total_time.count()) + " seconds.", false);


        // --- Collect Results for Saving ---
        // The 'documents' vector now contains the processed tokens in doc.tokens
        std::vector<std::vector<std::wstring>> results_for_csv;
        results_for_csv.reserve(documents.size());
        for (const auto &doc : documents) {
            results_for_csv.push_back(doc.tokens); // Collect the token vectors
        }


        // --- Sum Thread-Local Token Counts ---
        unsigned long total_tokens = 0;
        // Use std::accumulate for a concise sum (requires <numeric>)
        total_tokens = std::accumulate(thread_token_counts.begin(), thread_token_counts.end(), 0UL);
        /* Or using a simple loop:
        for(unsigned long count : thread_token_counts) {
            total_tokens += count;
        }
        */
        log_message("Total tokens processed (non-stopwords, stemmed): " + std::to_string(total_tokens), false);


        // --- Save Results and Performance Stats ---
        save_results_to_csv(results_for_csv, DATA_FILE);
        save_performance_stats(stats_name, total_tokens, documents.size(), numThreads, total_time);

    }
    catch (const std::exception &e) // Catch potential exceptions (e.g., from loading, saving)
    {
        log_message("CRITICAL ERROR in main processing block: " + std::string(e.what()), true);
        return 1; // Indicate failure
    }

    log_message("Processing pipeline finished successfully.", false);
    return 0; // Indicate success
}