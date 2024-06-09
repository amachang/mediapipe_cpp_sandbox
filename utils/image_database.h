#include <filesystem>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"

#include "opencv2/opencv.hpp"
#include <sqlite3.h>
#include <faiss/IndexFlat.h>

class ImageDatabase {
public:
    static absl::Status Initialize(const std::filesystem::path& model_path, uint64_t model_embedding_size);
    static ImageDatabase& GetInstance();
    absl::Status Insert(const std::filesystem::path& image_path);
    absl::Status Insert(const std::filesystem::path& image_path, const std::vector<float>& embedding);
    absl::StatusOr<bool> Exists(const std::filesystem::path& image_path);
    absl::StatusOr<float>SearchMaxSimilarity(const std::filesystem::path& image_path);
    absl::StatusOr<float> SearchMaxSimilarity(const std::vector<float>& embedding);
    absl::StatusOr<std::vector<std::pair<std::filesystem::path, float>>> Search(const std::filesystem::path& image_path, uint64_t top_k);
    absl::StatusOr<std::vector<std::pair<std::filesystem::path, float>>> Search(const std::vector<float>& embeddings, uint64_t top_k);
    absl::StatusOr<std::vector<float>> Embed(const std::filesystem::path& image_path);
    std::filesystem::path GetModelPath() const;
    static absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> LoadImageEmbedder(const std::filesystem::path& model_path);
    ImageDatabase(const ImageDatabase&) = delete;
    ImageDatabase& operator=(const ImageDatabase&) = delete;

private:
    const std::filesystem::path db_path_ = "unportable_image_database.db";

    std::mutex mutex_;
    std::optional<std::string> model_path_;
    std::optional<uint64_t> model_embedding_size_;
    std::optional<faiss::IndexFlatIP> faiss_index_;
    std::unordered_map<faiss::idx_t, uint64_t> faiss_id_to_db_id_;
    std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder> image_embedder_;
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db_;
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> insert_stmt_;
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> select_stmt_;
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> select_by_path_stmt_;

    static ImageDatabase& GetInstanceImpl(const std::optional<std::string>& model_path, std::optional<uint64_t> model_embedding_size);
    ImageDatabase(const std::optional<std::string>& model_path_opt, std::optional<uint64_t> model_embedding_size_opt);
};

