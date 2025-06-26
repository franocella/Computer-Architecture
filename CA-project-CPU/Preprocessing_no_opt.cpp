// System headers
#if defined(_WIN32)
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
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
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <limits>
#include <utility>

// Third-party library headers
#include "stemmer/english_stem.h"

// Constants
constexpr const char *LOG_FILE = "processing_log.csv";
constexpr const char *DATA_FILE = "processed_data.csv";
constexpr const char *STATS_FILENAME = "performance.csv";
constexpr int NUMTHREADS = 12;
constexpr int MAXWORKERS = 26;

// Forward declarations
struct Document;
std::wstring to_wstring(const std::string &str);
std::string to_string(const std::wstring &wstr);

// Data structures
struct Document
{
    std::wstring content;
    std::vector<std::wstring> tokens;
    int id;
};

// Global state (should be better encapsulated in a production system)
std::vector<Document> documents;
std::mutex log_mutex;
std::atomic<int> lastUpdatedIndex(0);
std::atomic<unsigned long> number_of_tokens(0);

// Constants for stopwords and regex pattern
const std::unordered_set<std::wstring> stopwords = {
    L"a", L"an", L"and", L"are", L"as", L"at", L"be", L"but", L"by",
    L"for", L"if", L"in", L"into", L"is", L"it", L"no", L"not", L"of",
    L"on", L"or", L"such", L"that", L"the", L"their", L"then", L"there",
    L"these", L"they", L"this", L"to", L"was", L"will", L"with"};

const std::wregex regex_pattern = std::wregex(
    LR"((?:[A-Z]\.+)|(?:\$?\d+(?:\.\d+)?%?)|(?:\b\w+@\w+\.\w+(?:\.\w+)*\b)|(?:https?://\S+)|(?:www\.\S+)|(?:\#\w+)|(?:\b\d{1,2}/\d{1,2}/\d{2,4}\b)|(?:\w+(?:-\w+)*))",
    std::regex_constants::icase | std::regex_constants::optimize);

// Utility functions
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

std::wstring to_wstring(const std::string &str)
{
#ifdef _WIN32
    if (str.empty())
        return L"";
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size_needed);
    return wstr;
#else
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(str);
#endif
}

std::string to_string(const std::wstring &wstr)
{
#ifdef _WIN32
    if (wstr.empty())
        return "";
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string str(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), (int)wstr.size(), &str[0], size_needed, NULL, NULL);
    return str;
#else
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.to_bytes(wstr);
#endif
}

// Core functionality

// Logs a message to the log file and to console
void log_message(const std::string &message, bool error = false)
{
    std::unique_lock<std::mutex> lock(log_mutex);
    std::ofstream log_file(LOG_FILE, std::ios_base::app | std::ios_base::out);

    std::stringstream ss;
    ss << "[" << get_current_time() << "] "
       << "[Thread " << std::this_thread::get_id() << "] "
       << (error ? "[ERROR] " : "[INFO] ") << message;
    const std::string final_msg = ss.str();

    if (log_file.is_open())
    {
        log_file << final_msg << "\n";
        log_file.flush();
    }
    else
    {
        std::cerr << "LOG ERROR: Failed to open log file\n";
    }

    if (error)
        std::cerr << final_msg << std::endl;
    else
        std::cout << final_msg << std::endl;
}

// Applies stemming to each token
void stem_tokens(std::vector<std::wstring> &tokens)
{
    stemming::english_stem<> stemmer;
    for (auto &token : tokens)
    {
        stemmer(token);
    }
}

// Filters out stopwords from the tokens vector between indices [start, end)
std::vector<std::wstring> filterStopwords(const std::vector<std::wstring> &tokens, size_t start, size_t end)
{
    std::vector<std::wstring> result;
    for (size_t i = start; i < end; ++i)
    {
        if (stopwords.find(tokens[i]) == stopwords.end())
            result.push_back(tokens[i]);
    }
    return result;
}

// OLD Tokenizes the input text using regex
std::vector<std::wstring> tokenize(const std::wstring &text)
{
    std::vector<std::wstring> tokens;
    auto begin = std::wsregex_token_iterator(text.begin(), text.end(), regex_pattern, 0);
    auto end = std::wsregex_token_iterator();
    for (auto it = begin; it != end; ++it)
    {
        tokens.push_back(it->str());
    }
    number_of_tokens.fetch_add(tokens.size());
    return tokens;
}

std::vector<std::wstring> processDocument(Document &doc)
{
    std::wstring content_copy = doc.content;

    std::transform(content_copy.begin(), content_copy.end(), content_copy.begin(),
                   [](wchar_t c)
                   { return std::towlower(c); });

    // Tokenization (with duplicates)
    std::vector<std::wstring> tokens = tokenize(content_copy);

    // Remove stopwords
    tokens = filterStopwords(tokens, 0, tokens.size());

    // Apply stemming
    stem_tokens(tokens);

    return tokens;
}

// Loads a TSV dataset from a file and returns a vector of Documents
std::vector<Document> load_tsv_dataset(const std::string &filename)
{
    std::vector<Document> loaded_docs;
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open dataset file");
    }

    // Handle BOM (Byte Order Mark)
    constexpr size_t BOM_SIZE = 3;
    char bom[BOM_SIZE];
    if (file.read(bom, BOM_SIZE) &&
        static_cast<unsigned char>(bom[0]) == 0xEF &&
        static_cast<unsigned char>(bom[1]) == 0xBB &&
        static_cast<unsigned char>(bom[2]) == 0xBF)
    {
        // BOM detected
    }
    else
    {
        file.seekg(0); // Reposition at the beginning
    }

    std::string line;
    size_t id = 0; // Prevent overflow for very large datasets
    size_t line_num = 0;
    while (std::getline(file, line))
    {
        line_num++;
        const size_t tab_pos = line.find('\t');

        if (tab_pos == std::string::npos || tab_pos >= line.size() - 1)
        {
            log_message("Skipping malformed line #" + std::to_string(line_num), false);
            continue;
        }

        Document doc;
        try
        {
            std::string content_str = line.substr(tab_pos + 1);
            if (content_str.empty())
            {
                throw std::runtime_error("No content");
            }
            doc.content = to_wstring(content_str);
        }
        catch (const std::exception &e)
        {
            log_message("Error converting line #" + std::to_string(line_num) + ": " + e.what(), true);
            continue;
        }
        doc.id = static_cast<int>(id++);
        loaded_docs.push_back(std::move(doc));
    }
    return loaded_docs;
}

// Task executed by each thread
void threadTask(int core_id, unsigned int num_threads)
{   
    while (true)
    {
        int index = lastUpdatedIndex.fetch_add(1);
        if (index >= static_cast<int>(documents.size()))
            break;
        
        Document &doc = documents[index];
        doc.tokens = processDocument(doc);
        doc.content.clear();
        doc.content.shrink_to_fit();
    }
}

// Saves the results (processed tokens) to a CSV file
void save_results_to_csv(const std::vector<std::vector<std::wstring>> &results,
                         const std::string &filename)
{
    try
    {
        std::ofstream csv_file(filename);
        if (!csv_file)
            throw std::runtime_error("Failed to create CSV file");

        csv_file << "doc_id,tokens\n";
        for (size_t i = 0; i < results.size(); ++i)
        {
            std::string token_line;
            for (const auto &token : results[i])
            {
                if (!token_line.empty())
                    token_line += ",";
                token_line += to_string(token);
            }
            csv_file << i << ",\"" << token_line << "\"\n";
        }
        log_message("Saved " + std::to_string(results.size()) + " records to " + filename);
    }
    catch (const std::exception &e)
    {
        log_message("CSV Error: " + std::string(e.what()), true);
    }
}

// Sets thread affinity (if supported by the platform)
void set_thread_affinity(int core_id)
{
#if defined(_WIN32)
    DWORD_PTR mask = 1ULL << core_id;
    if (!SetThreadAffinityMask(GetCurrentThread(), mask))
    {
        log_message("Failed to set affinity for core " + std::to_string(core_id), true);
    }
#else
    // On macOS or unsupported platforms, affinity is not supported: log a message
    log_message("Thread affinity is not supported on this platform", false);
#endif
}

void save_performance_stats(const std::string &filename, int num_tokens, int num_docs, int num_workers, std::chrono::duration<double> total_time)
{
    try
    {
        std::ofstream stats_file(filename, std::ios::app);
        if (!stats_file.is_open())
        {
            throw std::runtime_error("Failed to open/create performance stats file");
        }

        // Check if the file is empty to write the header
        std::ifstream check_file(filename);
        if (!check_file.is_open() || check_file.peek() == std::ifstream::traits_type::eof())
        {
            stats_file << "num_workers,num_tokens,num_docs,total_time\n";
        }
        check_file.close(); // Assicurati di chiudere il file di lettura

        stats_file << num_workers << ","
                   << num_tokens << ","
                   << num_docs << ","
                   << total_time.count() << "\n";
        stats_file.flush();
        stats_file.close(); // Assicurati di chiudere il file di scrittura
        log_message("Saved performance stats to " + filename);
    }
    catch (const std::exception &e)
    {
        log_message("Performance Stats Error: " + std::string(e.what()), true);
    }
}

int main(int argc, char *argv[])
{
    // If no parameters are provided, run in DEBUG mode using default thread count and default stats filename.
    unsigned int numThreads = NUMTHREADS;
    std::string stats_name = STATS_FILENAME;

    // if (argc < 2)
    // {
    //     log_message("DEBUG mode active: Using default thread count (" + std::to_string(NUMTHREADS) +
    //                 ") and default stats filename (" + STATS_FILENAME + ")");
    // }
    // else
    // {
    //     // Parse command-line arguments
    //     try
    //     {
    //         numThreads = static_cast<unsigned int>(std::stoi(argv[1]));
    //         if (numThreads < 1u)
    //             numThreads = 1u;
    //         else if (numThreads > static_cast<unsigned int>(MAXWORKERS))
    //             numThreads = static_cast<unsigned int>(MAXWORKERS);
    //     }
    //     catch (...)
    //     {
    //         std::cerr << "Invalid thread count, using default: " << NUMTHREADS << "\n";
    //     }

    //     if (argc >= 3)
    //     {
    //         stats_name = argv[2];
    //         size_t pos = stats_name.find_last_of('.');
    //         if (pos == std::string::npos)
    //             stats_name += ".csv";
    //         else
    //         {
    //             std::string ext = stats_name.substr(pos + 1);
    //             std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    //             if (ext != "csv")
    //             {
    //                 std::cerr << "Error: the file must have .csv extension\n";
    //                 return 1;
    //             }
    //         }
    //     }
    // }

    // Initialize log file (truncate)
    std::ofstream(LOG_FILE, std::ios::trunc);
    log_message("Starting document processing pipeline");

    try
    {
        // Load dataset
        log_message("Loading dataset...");
        documents = load_tsv_dataset("collection_shortened.tsv");
        log_message("Loaded " + std::to_string(documents.size()) + " documents");

        // Determine the number of available cores
        unsigned int numCores = std::thread::hardware_concurrency();
        if (numCores == 0)
        {
            numCores = 1;
            log_message("Could not detect core count, using single core");
        }
        log_message("System has " + std::to_string(numCores) + " available cores");

        // Create worker threads with affinity
        log_message("Initializing " + std::to_string(numThreads) + " worker threads");
        std::vector<std::thread> threads;
        threads.reserve(numThreads);

#ifdef _WIN32
        typedef HRESULT(WINAPI * SetThreadDescriptionType)(HANDLE, PCWSTR);
#endif
        std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < numThreads; ++i)
        {
            const int target_core = i % numCores;
            threads.emplace_back([target_core, numThreads]
                                 {
                set_thread_affinity(target_core);

#ifdef __linux__
                pthread_setname_np(pthread_self(), ("Worker_" + std::to_string(target_core)).c_str());
#elif defined(_WIN32)
                HMODULE kernel32 = GetModuleHandleW(L"kernel32");
                if (kernel32) {
                    auto SetThreadDescription = (SetThreadDescriptionType)GetProcAddress(kernel32, "SetThreadDescription");
                    if (SetThreadDescription) {
                        std::wstring thread_name = L"Worker_" + std::to_wstring(target_core);
                        SetThreadDescription(GetCurrentThread(), thread_name.c_str());
                    }
                }
#else
                pthread_setname_np(("Worker_" + std::to_string(target_core)).c_str());
#endif
                threadTask(target_core, numThreads); });
        }

        for (auto &t : threads)
        {
            if (t.joinable())
            {
                try
                {
                    t.join();
                }
                catch (const std::exception &e)
                {
                    log_message("Thread join failed: " + std::string(e.what()), true);
                }
            }
        }
        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = end_time - start_time;

        // Convert processed documents to vectors of tokens
        std::vector<std::vector<std::wstring>> results_for_csv;
        results_for_csv.reserve(documents.size());
        for (const auto &doc : documents)
        {
            results_for_csv.push_back(doc.tokens);

        }

        // Save results to CSV files
        save_results_to_csv(results_for_csv, DATA_FILE);
        save_performance_stats(stats_name, number_of_tokens.load(), documents.size(), numThreads, total_time);
    }
    catch (const std::exception &e)
    {
        log_message("Critical error: " + std::string(e.what()), true);
        return 1;
    }

    return 0;
}
