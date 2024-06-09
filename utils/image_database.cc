
#include "absl/status/status.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/image_frame.h"

#include "opencv2/opencv.hpp"
#include <sqlite3.h>
#include <faiss/IndexFlat.h>
#include "image_database.h"

absl::Status ImageDatabase::Initialize(const std::string& model_path, uint64_t model_embedding_size) {
    GetInstanceImpl(model_path, model_embedding_size);
}

ImageDatabase& ImageDatabase::GetInstance() {
    // When calling after Initialize, model_path and model_embedding_size are ignored because they are already set
    return GetInstanceImpl(std::nullopt, std::nullopt);
}

absl::Status ImageDatabase::Insert(const std::filesystem::path& image_path) {
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

absl::Status ImageDatabase::Insert(const std::filesystem::path& image_path, const std::vector<float>& embedding) {
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
    rc = sqlite3_bind_text(insert_stmt_.get(), 1, model_path_->c_str(), model_path_->size(), SQLITE_STATIC);
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
    faiss::idx_t faiss_id = faiss_index_->ntotal;
    faiss_index_->add(1, embedding.data());
    faiss_id_to_db_id_[faiss_id] = sqlite3_last_insert_rowid(db_.get());

    return absl::OkStatus();
}

absl::StatusOr<bool> ImageDatabase::Exists(const std::filesystem::path& image_path) {
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

    rc = sqlite3_bind_text(select_by_path_stmt_.get(), 1, model_path_->c_str(), model_path_->size(), SQLITE_STATIC);
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

absl::StatusOr<float> ImageDatabase::SearchMaxSimilarity(const std::filesystem::path& image_path) {
    absl::StatusOr<std::vector<float>> embedding_or_status = Embed(image_path);
    if (!embedding_or_status.ok()) {
        return absl::InternalError("Failed to embed image: " + std::string(embedding_or_status.status().message()));
    }
    const std::vector<float>& embedding = embedding_or_status.value();
    assert(embedding.size() == model_embedding_size_);

    return SearchMaxSimilarity(embedding);
}

absl::StatusOr<float> ImageDatabase::SearchMaxSimilarity(const std::vector<float>& embedding) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (faiss_index_->ntotal == 0) {
        return 0.0;
    }

    assert(embedding.size() == model_embedding_size_);

    float distances[1];
    faiss::idx_t result_ids[1];
    faiss_index_->search(1, embedding.data(), 1, distances, result_ids);

    return distances[0];
}

absl::StatusOr<std::vector<std::pair<std::filesystem::path, float>>> ImageDatabase::Search(const std::filesystem::path& image_path, uint64_t top_k) {
    absl::StatusOr<std::vector<float>> embedding_or_status = Embed(image_path);
    if (!embedding_or_status.ok()) {
        return absl::InternalError("Failed to embed image: " + std::string(embedding_or_status.status().message()));
    }
    const std::vector<float>& embedding = embedding_or_status.value();
    assert(embedding.size() == model_embedding_size_);

    return Search(embedding, top_k);
}

absl::StatusOr<std::vector<std::pair<std::filesystem::path, float>>> ImageDatabase::Search(const std::vector<float>& embeddings, uint64_t top_k) {
    std::lock_guard<std::mutex> lock(mutex_);

    assert(embeddings.size() == model_embedding_size_);

    float distances[top_k];
    faiss::idx_t result_ids[top_k];
    faiss_index_->search(1, embeddings.data(), top_k, distances, result_ids);

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
        assert(embedding_size == model_embedding_size_.value() * sizeof(float));

        const char* path = reinterpret_cast<const char*>(sqlite3_column_text(select_stmt_.get(), 1));
        std::string tmp_path_str(path);
        std::string path_str = tmp_path_str; // copy path
        results.push_back(std::make_pair(std::filesystem::path(path_str), distances[i]));
    }

    return results;

}

absl::StatusOr<std::vector<float>> ImageDatabase::Embed(const std::filesystem::path& image_path) {
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

absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> ImageDatabase::LoadImageEmbedder(const std::filesystem::path& model_path) {
    std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedderOptions> options = absl::make_unique<mediapipe::tasks::vision::image_embedder::ImageEmbedderOptions>();
    options->base_options.model_asset_path = model_path.string();
    options->embedder_options.l2_normalize = true;
   
    absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> image_embedder_or_status = mediapipe::tasks::vision::image_embedder::ImageEmbedder::Create(std::move(options));
    if (!image_embedder_or_status.ok()) {
        return absl::Status(absl::StatusCode::kInternal, "Failed to create image embedder");
    }
    return std::move(image_embedder_or_status.value());
}

ImageDatabase& ImageDatabase::GetInstanceImpl(const std::optional<std::string>& model_path, std::optional<uint64_t> model_embedding_size) {
    static ImageDatabase instance(model_path, model_embedding_size);
    return instance;
}

ImageDatabase::ImageDatabase(const std::optional<std::string>& model_path_opt, std::optional<uint64_t> model_embedding_size_opt) :
    model_path_(model_path_opt),
    model_embedding_size_(model_embedding_size_opt),
    faiss_index_(std::nullopt),
    faiss_id_to_db_id_(),
    image_embedder_(nullptr),
    db_(nullptr, sqlite3_close),
    insert_stmt_(nullptr, sqlite3_finalize),
    select_stmt_(nullptr, sqlite3_finalize),
    select_by_path_stmt_(nullptr, sqlite3_finalize) {

    if (!model_path_) {
        throw std::runtime_error("ImageDatabase: model_path uninitialized");
    }
    if (!model_embedding_size_) {
        throw std::runtime_error("ImageDatabase: model_embedding_size uninitialized");
    }

    faiss_index_.emplace(model_embedding_size_.value());

    // Load image embedder
    absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> image_embedder_or_status = LoadImageEmbedder(model_path_.value());
    if (!image_embedder_or_status.ok()) {
        throw std::runtime_error("Failed to load image embedder");
    }
    image_embedder_ = std::move(image_embedder_or_status.value());

    // Open database
    sqlite3* db = nullptr;
    int rc = sqlite3_open(db_path_.c_str(), &db);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to open database");
    }
    db_.reset(db);

    // Create table
    std::string create_table_query = "CREATE TABLE IF NOT EXISTS image (id INTEGER PRIMARY KEY, model TEXT, path TEXT, embedding BLOB, CONSTRAINT model_path UNIQUE (model, path))";
    char *error_message_cstr = nullptr;
    rc = sqlite3_exec(db_.get(), create_table_query.c_str(), nullptr, nullptr, &error_message_cstr);
    if (rc != SQLITE_OK) {
        assert(error_message_cstr != nullptr);
        std::string error_message("Failed to create table: " + std::string(error_message_cstr));
        sqlite3_free(error_message_cstr);
        throw std::runtime_error(error_message);
    }
    assert(error_message_cstr == nullptr);

    // Prepare insert statement
    std::string insert_query = "INSERT INTO image (model, path, embedding) VALUES (?, ?, ?)";
    sqlite3_stmt* insert_stmt = nullptr;
    rc = sqlite3_prepare_v2(db_.get(), insert_query.c_str(), insert_query.size(), &insert_stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare insert statement");
    }
    insert_stmt_.reset(insert_stmt);

    // Prepare select statement
    std::string select_query = "SELECT embedding, path FROM image WHERE id = ?";
    sqlite3_stmt* select_stmt = nullptr;
    rc = sqlite3_prepare_v2(db_.get(), select_query.c_str(), select_query.size(), &select_stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare select statement");
    }
    select_stmt_.reset(select_stmt);

    std::string select_by_path_query = "SELECT embedding FROM image WHERE model = ? AND path = ?";
    sqlite3_stmt* select_by_path_stmt = nullptr;
    rc = sqlite3_prepare_v2(db_.get(), select_by_path_query.c_str(), select_by_path_query.size(), &select_by_path_stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare select statement");
    }
    select_by_path_stmt_.reset(select_by_path_stmt);

    // Load embeddings from database
    std::string select_all_query = "SELECT id, path, embedding FROM image WHERE model = ?";
    sqlite3_stmt* select_all_stmt_ptr = nullptr;
    rc = sqlite3_prepare_v2(db_.get(), select_all_query.c_str(), select_all_query.size(), &select_all_stmt_ptr, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Failed to prepare select all statement");
    }
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> select_all_stmt(select_all_stmt_ptr, sqlite3_finalize);
    rc = sqlite3_bind_text(select_all_stmt.get(), 1, model_path_->c_str(), model_path_->size(), SQLITE_STATIC);

    while ((rc = sqlite3_step(select_all_stmt.get())) == SQLITE_ROW) {
        uint64_t db_id = sqlite3_column_int64(select_all_stmt.get(), 0);
        const char* path = reinterpret_cast<const char*>(sqlite3_column_text(select_all_stmt.get(), 1));
        const void* embedding = sqlite3_column_blob(select_all_stmt.get(), 2);
        int embedding_size = sqlite3_column_bytes(select_all_stmt.get(), 2);
        assert(embedding_size == model_embedding_size_.value() * sizeof(float));

        faiss::idx_t faiss_id = faiss_index_->ntotal;
        faiss_index_->add(1, reinterpret_cast<const float*>(embedding));
        faiss_id_to_db_id_[faiss_id] = db_id;
    }
    if (rc != SQLITE_DONE) {
        throw std::runtime_error("Failed to step select all statement");
    }
}

