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
#include <sqlite3.h>

class ImageDatabase {
public:
    ImageDatabase(const std::string& model_path, uint64_t model_embedding_size) :
        model_path_(model_path),
        model_embedding_size_(model_embedding_size),
        faiss_index_(model_embedding_size),
        faiss_id_to_db_id_(),
        image_embedder_(nullptr),
        db_(nullptr, sqlite3_close),
        insert_stmt_(nullptr, sqlite3_finalize),
        select_stmt_(nullptr, sqlite3_finalize),
        select_by_path_stmt_(nullptr, sqlite3_finalize) {
        }

    absl::Status Initialize() {
        std::lock_guard<std::mutex> lock(mutex_);

        // Load image embedder
        absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> image_embedder_or_status = LoadImageEmbedder(model_path_);
        if (!image_embedder_or_status.ok()) {
            return absl::InternalError("Failed to load image embedder");
        }
        image_embedder_ = std::move(image_embedder_or_status.value());

        // Open database
        sqlite3* db = nullptr;
        int rc = sqlite3_open(db_path_.c_str(), &db);
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to open database");
        }
        db_.reset(db);

        // Create table
        std::string create_table_query = "CREATE TABLE IF NOT EXISTS image (id INTEGER PRIMARY KEY, model TEXT, path TEXT, embedding BLOB, CONSTRAINT model_path UNIQUE (model, path))";
        char *error_message = nullptr;
        rc = sqlite3_exec(db_.get(), create_table_query.c_str(), nullptr, nullptr, &error_message);
        if (rc != SQLITE_OK) {
            assert(error_message != nullptr);
            absl::Status status = absl::InternalError("Failed to create table: " + std::string(error_message));
            sqlite3_free(error_message);
            return status;
        }
        assert(error_message == nullptr);

        // Prepare insert statement
        std::string insert_query = "INSERT INTO image (model, path, embedding) VALUES (?, ?, ?)";
        sqlite3_stmt* insert_stmt = nullptr;
        rc = sqlite3_prepare_v2(db_.get(), insert_query.c_str(), insert_query.size(), &insert_stmt, nullptr);
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to prepare insert statement");
        }
        insert_stmt_.reset(insert_stmt);

        // Prepare select statement
        std::string select_query = "SELECT embedding, path FROM image WHERE id = ?";
        sqlite3_stmt* select_stmt = nullptr;
        rc = sqlite3_prepare_v2(db_.get(), select_query.c_str(), select_query.size(), &select_stmt, nullptr);
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to prepare select statement");
        }
        select_stmt_.reset(select_stmt);

        std::string select_by_path_query = "SELECT embedding FROM image WHERE model = ? AND path = ?";
        sqlite3_stmt* select_by_path_stmt = nullptr;
        rc = sqlite3_prepare_v2(db_.get(), select_by_path_query.c_str(), select_by_path_query.size(), &select_by_path_stmt, nullptr);
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to prepare select statement");
        }
        select_by_path_stmt_.reset(select_by_path_stmt);

        // Load embeddings from database
        std::string select_all_query = "SELECT id, path, embedding FROM image WHERE model = ?";
        sqlite3_stmt* select_all_stmt_ptr = nullptr;
        rc = sqlite3_prepare_v2(db_.get(), select_all_query.c_str(), select_all_query.size(), &select_all_stmt_ptr, nullptr);
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to prepare select all statement");
        }
        std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> select_all_stmt(select_all_stmt_ptr, sqlite3_finalize);
        rc = sqlite3_bind_text(select_all_stmt.get(), 1, model_path_.c_str(), model_path_.size(), SQLITE_STATIC);

        while ((rc = sqlite3_step(select_all_stmt.get())) == SQLITE_ROW) {
            uint64_t db_id = sqlite3_column_int64(select_all_stmt.get(), 0);
            const char* path = reinterpret_cast<const char*>(sqlite3_column_text(select_all_stmt.get(), 1));
            const void* embedding = sqlite3_column_blob(select_all_stmt.get(), 2);
            int embedding_size = sqlite3_column_bytes(select_all_stmt.get(), 2);
            assert(embedding_size == model_embedding_size_ * sizeof(float));

            faiss::idx_t faiss_id = faiss_index_.ntotal;
            faiss_index_.add(1, reinterpret_cast<const float*>(embedding));
            faiss_id_to_db_id_[faiss_id] = db_id;
        }
        if (rc != SQLITE_DONE) {
            return absl::InternalError("Failed to step select all statement");
        }

        return absl::OkStatus();
    }

    absl::Status Insert(const std::filesystem::path& image_path) {
        if (!std::filesystem::exists(image_path)) {
            return absl::InternalError("Image does not exist: " + image_path.string());
        }

        absl::StatusOr<std::vector<float>> embedding_or_status = Embed(image_path);
        if (!embedding_or_status.ok()) {
            return absl::InternalError("Failed to embed image: " + std::string(embedding_or_status.status().message()));
        }
        const std::vector<float>& embedding = embedding_or_status.value();
        assert(embedding.size() == model_embedding_size_);

        return Insert(image_path, embedding);
    }

    absl::Status Insert(const std::filesystem::path& image_path, const std::vector<float>& embedding) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::string image_path_str = image_path.string();

        int rc = sqlite3_reset(insert_stmt_.get());
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to reset select statement");
        }
        rc = sqlite3_clear_bindings(insert_stmt_.get());
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to clear bindings");
        }

        // Insert into database
        rc = sqlite3_bind_text(insert_stmt_.get(), 1, model_path_.c_str(), model_path_.size(), SQLITE_STATIC);
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to bind model path");
        }
        rc = sqlite3_bind_text(insert_stmt_.get(), 2, image_path_str.c_str(), image_path_str.size(), SQLITE_STATIC);
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to bind image path");
        }
        rc = sqlite3_bind_blob(insert_stmt_.get(), 3, embedding.data(), embedding.size() * sizeof(float), SQLITE_STATIC);
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to bind embedding");
        }
        rc = sqlite3_step(insert_stmt_.get());
        if (rc != SQLITE_DONE) {
            return absl::InternalError("Failed to insert image");
        }

        // Update faiss index
        faiss::idx_t faiss_id = faiss_index_.ntotal;
        faiss_index_.add(1, embedding.data());
        faiss_id_to_db_id_[faiss_id] = sqlite3_last_insert_rowid(db_.get());

        return absl::OkStatus();
    }

    absl::StatusOr<bool> Exists(const std::filesystem::path& image_path) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::string image_path_str = image_path.string();

        int rc = sqlite3_reset(select_by_path_stmt_.get());
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to reset select statement");
        }
        rc = sqlite3_clear_bindings(select_by_path_stmt_.get());
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to clear bindings");
        }

        rc = sqlite3_bind_text(select_by_path_stmt_.get(), 1, model_path_.c_str(), model_path_.size(), SQLITE_STATIC);
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to bind model path");
        }
        rc = sqlite3_bind_text(select_by_path_stmt_.get(), 2, image_path_str.c_str(), image_path_str.size(), SQLITE_STATIC);
        if (rc != SQLITE_OK) {
            return absl::InternalError("Failed to bind image path");
        }

        rc = sqlite3_step(select_by_path_stmt_.get());
        if (rc == SQLITE_ROW) {
            return true;
        } else if (rc == SQLITE_DONE) {
            return false;
        } else {
            return absl::InternalError("Failed to step select statement");
        }
    }

    absl::StatusOr<float>SearchMaxSimilarity(const std::filesystem::path& image_path) {
        absl::StatusOr<std::vector<float>> embedding_or_status = Embed(image_path);
        if (!embedding_or_status.ok()) {
            return absl::InternalError("Failed to embed image: " + std::string(embedding_or_status.status().message()));
        }
        const std::vector<float>& embedding = embedding_or_status.value();
        assert(embedding.size() == model_embedding_size_);

        return SearchMaxSimilarity(embedding);
    }

    absl::StatusOr<float> SearchMaxSimilarity(const std::vector<float>& embedding) {
        std::lock_guard<std::mutex> lock(mutex_);

        assert(embedding.size() == model_embedding_size_);

        float distances[1];
        faiss::idx_t result_ids[1];
        faiss_index_.search(1, embedding.data(), 1, distances, result_ids);

        return distances[0];
    }

    absl::StatusOr<std::vector<std::pair<std::filesystem::path, float>>> Search(const std::filesystem::path& image_path, uint64_t top_k) {
        absl::StatusOr<std::vector<float>> embedding_or_status = Embed(image_path);
        if (!embedding_or_status.ok()) {
            return absl::InternalError("Failed to embed image: " + std::string(embedding_or_status.status().message()));
        }
        const std::vector<float>& embedding = embedding_or_status.value();
        assert(embedding.size() == model_embedding_size_);

        return Search(embedding, top_k);
    }

    absl::StatusOr<std::vector<std::pair<std::filesystem::path, float>>> Search(const std::vector<float>& embeddings, uint64_t top_k) {
        std::lock_guard<std::mutex> lock(mutex_);

        assert(embeddings.size() == model_embedding_size_);

        float distances[top_k];
        faiss::idx_t result_ids[top_k];
        faiss_index_.search(1, embeddings.data(), top_k, distances, result_ids);

        std::vector<std::pair<std::filesystem::path, float>> results;
        for (uint64_t i = 0; i < top_k; i++) {
            faiss::idx_t faiss_id = result_ids[i];
            auto it = faiss_id_to_db_id_.find(faiss_id);
            if (it == faiss_id_to_db_id_.end()) {
                continue;
            }
            uint64_t db_id = it->second;

            int rc = sqlite3_reset(select_stmt_.get());
            if (rc != SQLITE_OK) {
                return absl::InternalError("Failed to reset select statement");
            }
            rc = sqlite3_clear_bindings(select_stmt_.get());
            if (rc != SQLITE_OK) {
                return absl::InternalError("Failed to clear bindings");
            }

            rc = sqlite3_bind_int64(select_stmt_.get(), 1, db_id);
            if (rc != SQLITE_OK) {
                return absl::InternalError("Failed to bind db id");
            }

            rc = sqlite3_step(select_stmt_.get());
            if (rc != SQLITE_ROW) {
                return absl::InternalError("Failed to step select statement");
            }

            const void* embedding = sqlite3_column_blob(select_stmt_.get(), 0);
            int embedding_size = sqlite3_column_bytes(select_stmt_.get(), 0);
            assert(embedding_size == model_embedding_size_ * sizeof(float));

            const char* path = reinterpret_cast<const char*>(sqlite3_column_text(select_stmt_.get(), 1));
            std::string tmp_path_str(path);
            std::string path_str = tmp_path_str; // copy path
            results.push_back(std::make_pair(std::filesystem::path(path_str), distances[i]));
        }

        return results;
    
    }

    absl::StatusOr<std::vector<float>> Embed(const std::filesystem::path& image_path) {
        cv::Mat image_mat = cv::imread(image_path.string());
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

        const absl::StatusOr<mediapipe::tasks::components::containers::EmbeddingResult> result_or_status = image_embedder_->Embed(image);
        if (!result_or_status.ok()) {
            return absl::Status(absl::StatusCode::kInternal, "Failed to embed image");
        }

        const mediapipe::tasks::components::containers::EmbeddingResult& result = result_or_status.value();
        const std::vector<float> &embedding = result.embeddings[0].float_embedding;
        return embedding;
    }

    static absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> LoadImageEmbedder(const std::filesystem::path& model_path) {
        std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedderOptions> options = absl::make_unique<mediapipe::tasks::vision::image_embedder::ImageEmbedderOptions>();
        options->base_options.model_asset_path = model_path.string();
        options->embedder_options.l2_normalize = true;
       
        absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> image_embedder_or_status = mediapipe::tasks::vision::image_embedder::ImageEmbedder::Create(std::move(options));
        if (!image_embedder_or_status.ok()) {
            return absl::Status(absl::StatusCode::kInternal, "Failed to create image embedder");
        }
        return std::move(image_embedder_or_status.value());
    }

private:
    const std::filesystem::path db_path_ = "unportable_image_database.db";

    std::mutex mutex_;
    std::string model_path_;
    uint64_t model_embedding_size_;
    faiss::IndexFlatIP faiss_index_;
    std::unordered_map<faiss::idx_t, uint64_t> faiss_id_to_db_id_;
    std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder> image_embedder_;
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db_;
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> insert_stmt_;
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> select_stmt_;
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> select_by_path_stmt_;
};

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

