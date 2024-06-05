#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <thread>

#include <faiss/IndexFlat.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "mediapipe/util/frame_buffer/frame_buffer_util.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/image_frame.h"

#include "opencv2/opencv.hpp"
#include "utils/image_database.h"

const std::filesystem::path kModelPath = "mobilenet_v3_large.tflite";
const uint64_t kModelEmbeddingSize = 1280;

ABSL_FLAG(std::string, dir, "", "Path to the image store directory");

int main(int argc, char** argv) {
    // prerequisites
    assert(sizeof(float) == 4); // float should be 4 bytes. prerequisite for mediapipe
    assert(std::filesystem::exists(kModelPath));

    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);

    const std::string dir_str = absl::GetFlag(FLAGS_dir);
    if (dir_str.empty()) {
        std::cerr << "Please provide a valid directory path" << std::endl;
        return EXIT_FAILURE;
    }
    std::filesystem::path dir(dir_str);

    std::cout << "Collecting image paths: " << dir_str << std::endl;
    std::unordered_set<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"};
    std::vector<std::filesystem::path> image_paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir_str)) {
        if (valid_extensions.find(entry.path().extension().string()) == valid_extensions.end()) {
            continue;
        }
        image_paths.push_back(entry.path());
    }
    uint64_t n_images = image_paths.size();
    std::cout << "Found " << n_images << " images" << std::endl;

    // inner product with normalized vectors is equivalent to cosine similarity
    ImageDatabase image_database(kModelPath, kModelEmbeddingSize);
    absl::Status status = image_database.Initialize();
    if (!status.ok()) {
        LOG(ERROR) << "Failed to initialize image database: " << status.message();
        return EXIT_FAILURE;
    }

    std::cout << "Embedding images to vectors" << std::endl;
    uint64_t n_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (uint64_t thread_index = 0; thread_index < n_threads; thread_index++) {
        threads.push_back(std::thread([&image_database, &image_paths, n_images, n_threads, thread_index]() {
            LOG(INFO) << "Thread started: " << std::this_thread::get_id();
            uint64_t n_thread_assigned = n_images / n_threads;
            for (uint64_t i = thread_index; i < n_images; i += n_threads) {
                uint64_t n_thread_processed = (i - thread_index) / n_threads;
                if (n_thread_processed % 100 == 0) {
                    LOG(INFO) << "Thread: " << std::this_thread::get_id() << " processed " << n_thread_processed << "/" << n_thread_assigned << " images";
                }
                const std::filesystem::path& image_path = image_paths[i];
                absl::StatusOr<bool> exists_or_status = image_database.Exists(image_path);
                if (!exists_or_status.ok()) {
                    LOG(ERROR) << "Failed to check if image exists: " << image_path << " (" << exists_or_status.status().message() << ")";
                    continue;
                }
                if (exists_or_status.value()) {
                    LOG(INFO) << "Image already exists: " << image_path;
                    continue;
                }
                absl::Status status = image_database.Insert(image_path);
                if (!status.ok()) {
                    LOG(ERROR) << "Failed to insert image: " << image_path << " (" << status.message() << ")";
                    continue;
                }
            }
        }));
    }

    // join threads
    for (std::thread& thread : threads) {
        thread.join();
        std::cout << "Joining thread: " << thread.get_id() << std::endl;
    }
    std::cout << "Finished embedding images" << std::endl;

    uint64_t top_k = 5;
    float distances[top_k];
    faiss::idx_t result_ids[top_k];

    // repr
    while (true) {
        std::string query_image_path_str;
        std::cout << "Enter query image path: ";
        std::cin >> query_image_path_str;
        if (query_image_path_str == "exit") {
            break;
        }
        std::filesystem::path query_image_path(query_image_path_str);
        if (!std::filesystem::exists(query_image_path)) {
            std::cout << "Image does not exist: " << query_image_path << std::endl;
            continue;
        }

        absl::StatusOr<std::vector<std::pair<std::filesystem::path, float>>> search_result_or_status = image_database.Search(query_image_path, top_k);
        if (!search_result_or_status.ok()) {
            std::cout << "Failed to search image: " << search_result_or_status.status().message() << std::endl;
            continue;
        }
        std::vector<std::pair<std::filesystem::path, float>>& search_result = search_result_or_status.value();

        cv::imshow("Query Image", cv::imread(query_image_path.string()));
        for (uint64_t i = 0; i < search_result.size(); i++) {
            std::pair<std::filesystem::path, float> result = search_result[i];
            cv::imshow("Result Image " + std::to_string(i), cv::imread(result.first.string()));
            std::cout << "Result: " << result.first << " (distance: " << result.second << ")" << std::endl;
        }
        cv::waitKey(1);
    }

    return EXIT_SUCCESS;
}

