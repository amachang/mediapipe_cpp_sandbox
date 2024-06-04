#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <filesystem>

#include "opencv2/opencv.hpp"

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "mediapipe/util/frame_buffer/frame_buffer_util.h"
#include "mediapipe/tasks/cc/vision/image_embedder/image_embedder.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/image_frame.h"

ABSL_FLAG(std::string, image1_path, "", "image1 path");
ABSL_FLAG(std::string, image2_path, "", "model path");

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    absl::ParseCommandLine(argc, argv);

    std::string image1_path_str = absl::GetFlag(FLAGS_image1_path);
    if (image1_path_str.empty()) {
        LOG(ERROR) << "Please specify image1 path with --image1_path";
        return EXIT_FAILURE;
    }
    std::filesystem::path image1_path(image1_path_str);
    if (!std::filesystem::exists(image1_path)) {
        LOG(ERROR) << "Image1 path does not exist: " << image1_path;
        return EXIT_FAILURE;
    }

    std::string image2_path_str = absl::GetFlag(FLAGS_image2_path);
    if (image2_path_str.empty()) {
        LOG(ERROR) << "Please specify image2 path with --image2_path";
        return EXIT_FAILURE;
    }
    std::filesystem::path image2_path(image2_path_str);
    if (!std::filesystem::exists(image2_path)) {
        LOG(ERROR) << "Image2 path does not exist: " << image2_path;
        return EXIT_FAILURE;
    }

    // Initialization
    std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedderOptions> options = absl::make_unique<mediapipe::tasks::vision::image_embedder::ImageEmbedderOptions>();
    options->base_options.model_asset_path = "mobilenet_v3_large.tflite";
    options->embedder_options.l2_normalize = true;
   
    absl::StatusOr<std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder>> image_embedder_or_status = mediapipe::tasks::vision::image_embedder::ImageEmbedder::Create(std::move(options));
    if (!image_embedder_or_status.ok()) {
        LOG(ERROR) << "Failed to create image embedder: " << image_embedder_or_status.status();
        return EXIT_FAILURE;
    }
    std::unique_ptr<mediapipe::tasks::vision::image_embedder::ImageEmbedder> image_embedder = std::move(image_embedder_or_status.value());

    // Load images with openCV
    cv::Mat image_mat1 = cv::imread(image1_path.string());
    // cv::imshow("image1", image_mat1);
    // cv::waitKey(0);

    cv::Mat image_mat2 = cv::imread(image2_path.string());
    // cv::imshow("image2", image_mat2);
    // cv::waitKey(0);

    // Convert openCV images to mediapipe ImageFrame
    std::shared_ptr<mediapipe::ImageFrame> image_frame1 = std::make_shared<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, image_mat1.cols, image_mat1.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    std::shared_ptr<mediapipe::ImageFrame> image_frame2 = std::make_shared<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, image_mat2.cols, image_mat2.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);

    cv::Mat image_mat_view1 = mediapipe::formats::MatView(image_frame1.get());
    cv::Mat image_mat_view2 = mediapipe::formats::MatView(image_frame2.get());

    cv::cvtColor(image_mat1, image_mat_view1, cv::COLOR_BGR2RGB);
    cv::cvtColor(image_mat2, image_mat_view2, cv::COLOR_BGR2RGB);

    mediapipe::Image image1(image_frame1);
    mediapipe::Image image2(image_frame2);

    // Run inference on two images.
    const absl::StatusOr<mediapipe::tasks::components::containers::EmbeddingResult> result1_or_status = image_embedder->Embed(image1);
    if (!result1_or_status.ok()) {
        LOG(ERROR) << "Failed to embed image: " << result1_or_status.status();
        return EXIT_FAILURE;
    }
    std::cout << "Image1 embedded: " << result1_or_status.value().embeddings.size() << std::endl;

    const absl::StatusOr<mediapipe::tasks::components::containers::EmbeddingResult> result2_or_status = image_embedder->Embed(image2);
    if (!result2_or_status.ok()) {
        LOG(ERROR) << "Failed to embed image: " << result2_or_status.status();
        return EXIT_FAILURE;
    }
    std::cout << "Image2 embedded: " << result2_or_status.value().embeddings.size() << std::endl;

    const mediapipe::tasks::components::containers::EmbeddingResult& result1 = result1_or_status.value();
    const mediapipe::tasks::components::containers::EmbeddingResult& result2 = result2_or_status.value();

    // Compute cosine similarity.
    absl::StatusOr<double> similarity_or_status = mediapipe::tasks::vision::image_embedder::ImageEmbedder::CosineSimilarity(result1.embeddings[0], result2.embeddings[0]);
    if (!similarity_or_status.ok()) {
        LOG(ERROR) << "Failed to compute cosine similarity: " << similarity_or_status.status();
        return EXIT_FAILURE;
    }
    double similarity = similarity_or_status.value();

    std::cout << "Cosine similarity: " << similarity << std::endl;

    return 0;
}

