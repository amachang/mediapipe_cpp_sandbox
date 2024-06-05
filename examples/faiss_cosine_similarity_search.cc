#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <filesystem>
#include <chrono>

#include <faiss/IndexFlat.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "mediapipe/util/frame_buffer/frame_buffer_util.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/image_frame.h"

#include "opencv2/opencv.hpp"

const std::filesystem::path kModelPath = "mobilenet_v3_large.tflite";
const uint64_t kModelEmbeddingSize = 1280;

ABSL_FLAG(std::string, dir, "", "Path to the image store directory");

absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> LoadImageEmbedder(const std::filesystem::path& model_path) {
    std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedderOptions> options = absl::make_unique<mediapipe::tasks::vision::image_embedder::ImageEmbedderOptions>();
    options->base_options.model_asset_path = model_path.string();
    options->embedder_options.l2_normalize = true;
   
    absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> image_embedder_or_status = mediapipe::tasks::vision::image_embedder::ImageEmbedder::Create(std::move(options));
    if (!image_embedder_or_status.ok()) {
        return absl::Status(absl::StatusCode::kInternal, "Failed to create image embedder");
    }
    return std::move(image_embedder_or_status.value());
}

absl::StatusOr<std::vector<float>> Embed(const std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>& image_embedder, const std::string& image_path) {
    cv::Mat image_mat = cv::imread(image_path);
    if (image_mat.empty()) {
        return absl::Status(absl::StatusCode::kInternal, "Failed to read image");
    }
    if (image_mat.channels() != 3) {
        return absl::Status(absl::StatusCode::kInternal, "Currently only RGB images are supported");
    }

    std::shared_ptr<mediapipe::ImageFrame> image_frame = std::make_shared<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, image_mat.cols, image_mat.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat image_mat_view = mediapipe::formats::MatView(image_frame.get());
    cv::cvtColor(image_mat, image_mat_view, cv::COLOR_BGR2RGB);
    mediapipe::Image image(image_frame);

    const absl::StatusOr<mediapipe::tasks::components::containers::EmbeddingResult> result_or_status = image_embedder->Embed(image);
    if (!result_or_status.ok()) {
        return absl::Status(absl::StatusCode::kInternal, "Failed to embed image");
    }

    const mediapipe::tasks::components::containers::EmbeddingResult& result = result_or_status.value();
    const std::vector<float> &embedding = result.embeddings[0].float_embedding;
    return embedding;
}

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

    // Load image embedder
    absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> image_embedder_or_status = LoadImageEmbedder(kModelPath);
    if (!image_embedder_or_status.ok()) {
        LOG(ERROR) << "Failed to load image embedder: " << image_embedder_or_status.status().message();
        return EXIT_FAILURE;
    }
    std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder> image_embedder = std::move(image_embedder_or_status.value());

    std::cout << "Collecting image paths: " << dir_str << std::endl;
    std::unordered_set<std::string> valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"};
    std::vector<std::string> image_paths;
    for (const auto& entry : std::filesystem::directory_iterator(dir_str)) {
        if (valid_extensions.find(entry.path().extension().string()) == valid_extensions.end()) {
            continue;
        }
        image_paths.push_back(entry.path().string());
    }
    uint64_t n_images = image_paths.size();
    std::cout << "Found " << n_images << " images" << std::endl;

    // inner product with normalized vectors is equivalent to cosine similarity
    faiss::IndexFlatIP faiss_index(kModelEmbeddingSize);
    std::unique_ptr<float[]> vectors(new float[kModelEmbeddingSize * n_images]);
    std::unordered_map<uint64_t, std::string> faiss_id_to_image_path;

    std::cout << "Embedding images to vectors" << std::endl;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    uint64_t n_embedded = 0;
    uint64_t n_errors = 0;
    for (uint64_t i = 0; i < n_images; i++) {
        uint64_t n_processed = n_embedded + n_errors;
        if (n_processed > 0 && n_processed % 100 == 0) {
            std::chrono::steady_clock::time_point current_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(current_time - start_time);

            assert(n_processed > 0);
            double eta = elapsed_seconds.count() / n_processed * (n_images - n_processed);

            std::cout << "Embedded " << n_processed << "/" << n_images << " images (" << elapsed_seconds.count() << "s, ETA: " << eta << "s)" << std::endl;
        }

        const std::string& image_path = image_paths[i];

        absl::StatusOr<std::vector<float>> embedding_or_status = Embed(image_embedder, image_path);
        if (!embedding_or_status.ok()) {
            LOG(ERROR) << "Failed to embed image: " << image_path << " (" << embedding_or_status.status().message() << ")";
            n_errors++;
            continue;
        }
        const std::vector<float>&& embedding = std::move(embedding_or_status.value());

        assert(embedding.size() == kModelEmbeddingSize);

        float *to_ptr = vectors.get() + kModelEmbeddingSize * n_embedded;
        const float *from_ptr = embedding.data();
        memcpy(to_ptr, from_ptr, kModelEmbeddingSize * sizeof(float));
        faiss::idx_t faiss_id = n_embedded;
        faiss_id_to_image_path[faiss_id] = image_path;

        n_embedded++;
    }
    std::cout << "Embedded " << n_embedded << " images" << std::endl;

    std::cout << "Building faiss index" << std::endl;
    faiss_index.add(n_embedded, vectors.get());
    std::cout << "Faiss index built" << std::endl;

    uint64_t top_k = 5;
    float distances[top_k * kModelEmbeddingSize];
    faiss::idx_t result_ids[top_k];

    // repr
    while (true) {
        std::string query_image_path;
        std::cout << "Enter query image path: ";
        std::cin >> query_image_path;
        if (query_image_path == "exit") {
            break;
        }
        absl::StatusOr<std::vector<float>> query_embedding_or_status = Embed(image_embedder, query_image_path);
        if (!query_embedding_or_status.ok()) {
            LOG(ERROR) << "Failed to embed query image: " << query_image_path << " (" << query_embedding_or_status.status().message() << ")";
            continue;
        }
        const std::vector<float>& query_embedding = query_embedding_or_status.value();
        assert(query_embedding.size() == kModelEmbeddingSize);
        faiss_index.search(1, query_embedding.data(), top_k, distances, result_ids);

        cv::imshow("Query Image", cv::imread(query_image_path));
        for (uint64_t i = 0; i < top_k; i++) {
            const std::string& result_image_path = faiss_id_to_image_path[result_ids[i]];
            cv::imshow("Result Image " + std::to_string(i), cv::imread(result_image_path));
            std::cout << "Result " << i << ": " << result_image_path << " (distance: " << distances[i] << ")" << std::endl;
        }
        cv::waitKey(0);
    }

    return EXIT_SUCCESS;
}

