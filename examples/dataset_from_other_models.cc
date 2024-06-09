#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <filesystem>
#include <chrono>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <algorithm>
#include <tuple>

#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "glog/logging.h"

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"

#include <inja/inja.hpp>
#include <sqlite3.h>
#include "utils/image_database.h"

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

int window_width = 800;

ABSL_FLAG(std::string, video_dir, "", "video dir");
ABSL_FLAG(std::string, model_path, "", "model path");
ABSL_FLAG(std::string, output_dir, "", "output dir");

ABSL_FLAG(uint, model_input_width, 300, "video input width");
ABSL_FLAG(uint, model_input_height, 300, "video input height");
ABSL_FLAG(std::string, model_input_scale_mode, "FIT", "video input scale mode");
ABSL_FLAG(double, capture_period, 5.0, "capture period");
ABSL_FLAG(double, similarity_threshold, 0.75, "similarity threshold");

std::string score_to_label(double score) {
    int i_score = static_cast<int>(score * 10);
    std::string score_label = std::to_string(i_score);
    if (i_score > 100) {
        score_label = "99";
    }
    // pad with 0
    int pad_len = std::max(0, 2 - static_cast<int>(score_label.size()));
    score_label = std::string(pad_len, '0') + score_label;
    return score_label;
}

absl::Status RunMediapipe(
        const std::filesystem::path& video_path, 
        const std::filesystem::path& model_path,
        const std::filesystem::path& output_dir,
        const double capture_period,
        const std::pair<uint, uint> model_input_size,
        const std::string& model_input_scale_mode,
        const double similarity_threshold
        ) {

    uint model_input_width = model_input_size.first;
    uint model_input_height = model_input_size.second;
    uint64_t capture_period_us = static_cast<uint64_t>(capture_period * 1e6);

    inja::Environment env;
    inja::Template calculator_graph_template = env.parse(R"pb(
        output_stream: "output_stream"

        # Input video
        node {
            calculator: "VideoDecoderCalculator"
            input_side_packet: "INPUT_FILE_PATH:input_video_path"
            output_stream: "VIDEO:input_stream"
        }

        node {
            calculator: "PacketThinnerCalculator"
            input_stream: "input_stream"
            output_stream: "downsampled_input_image"
            node_options: {
                [type.googleapis.com/mediapipe.PacketThinnerCalculatorOptions] {
                    thinner_type: ASYNC
                    period: {{ capture_period_us }}
                }
            }
        }

        # Detect face
        node {
            calculator: "ToImageCalculator"
            input_stream: "IMAGE:downsampled_input_image"
            output_stream: "IMAGE:used_by_face_detector_image"
        }
        node {
            calculator: "mediapipe.tasks.vision.face_detector.FaceDetectorGraph"
            input_stream: "IMAGE:used_by_face_detector_image"
            output_stream: "EXPANDED_FACE_RECTS:face_rects"
            node_options: {
                [type.googleapis.com/mediapipe.tasks.vision.face_detector.proto.FaceDetectorGraphOptions] {
                    base_options {
                        model_asset {
                            file_name: "blaze_face_short_range.tflite"
                        }
                    }
                    min_detection_confidence: 0.5
                    num_faces: 1
                }
            }
        }
        node {
            calculator: "SplitNormalizedRectVectorCalculator"
            input_stream: "face_rects"
            output_stream: "face_rect"
            node_options: {
                [type.googleapis.com/mediapipe.SplitVectorCalculatorOptions] {
                    ranges: { begin: 0 end: 1 }
                    element_only: true
                }
            }
        }

        # Crop image
        node {
            calculator: "ImageCroppingCalculator"
            input_stream: "IMAGE:downsampled_input_image"
            input_stream: "NORM_RECT:face_rect"
            output_stream: "IMAGE:cropped_image_stream"
        }
        node: {
            calculator: "ImageTransformationCalculator"
            input_stream: "IMAGE:cropped_image_stream"
            output_stream: "IMAGE:square_input_image"
            node_options: {
                [type.googleapis.com/mediapipe.ImageTransformationCalculatorOptions] {
                    output_width: {{ model_input_width }}
                    output_height: {{ model_input_height }}
                    scale_mode: {{ model_input_scale_mode }}
                }
            }
        }

        node {
            calculator: "ToImageCalculator"
            input_stream: "IMAGE:square_input_image"
            output_stream: "IMAGE:image_used_by_vision_tasks"
        }

        # Embed image
        node {
            calculator: "mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph"
            input_stream: "IMAGE:image_used_by_vision_tasks"
            output_stream: "EMBEDDINGS:embeddings"
            node_options {
                [type.googleapis.com/mediapipe.tasks.vision.image_embedder.proto.ImageEmbedderGraphOptions] {
                    base_options {
                        model_asset {
                            file_name: "{{ image_database_model_path }}"
                        }
                    }
                    embedder_options {
                        l2_normalize: true
                    }
                }
            }
        }

        # Embedding similarity gate
        node {
            calculator: "SimilarImageGateCalculator"
            input_stream: "EMBEDDINGS:embeddings"
            input_stream: "image_used_by_vision_tasks"
            input_stream: "cropped_image_stream"
            output_stream: "EMBEDDINGS:gated_embeddings"
            output_stream: "gated_image_used_by_vision_tasks"
            output_stream: "gated_cropped_image_stream"
            node_options: {
                [type.googleapis.com/SimilarImageGateCalculatorOptions] {
                    similarity_threshold: {{ similarity_threshold }}
                }
            }
        }

        # Classify image
        node {
            calculator: "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph"
            input_stream: "IMAGE:gated_image_used_by_vision_tasks"
            output_stream: "CLASSIFICATIONS:gated_classifications"
            node_options: {
                [type.googleapis.com/mediapipe.tasks.vision.image_classifier.proto.ImageClassifierGraphOptions] {
                    base_options: {
                        model_asset: {
                            file_name: "{{ model_path }}" # TODO escape path
                        }
                    }
                    classifier_options: {
                        max_results: 1
                    }
                }
            }
        }

        # To callback
        node {
            calculator: "ToImageCalculator"
            input_stream: "IMAGE:gated_cropped_image_stream"
            output_stream: "IMAGE:gated_type_converted_image"
        }
        node {
            calculator: "DatasetCandidateGateCalculator"
            input_stream: "EMBEDDINGS:gated_embeddings"
            input_stream: "CLASSIFICATIONS:gated_classifications"
            input_stream: "IMAGE:gated_type_converted_image"
            output_stream: "LABEL_SCORE_EMBEDDING_IMAGE:output_stream"
        }
    )pb");

    ImageDatabase &image_database = ImageDatabase::GetInstance();
    std::filesystem::path image_database_model_path = image_database.GetModelPath();

    std::string calculator_graph_str = env.render(calculator_graph_template, {
            {"model_path", model_path.string()},
            {"capture_period_us", std::to_string(capture_period_us)},
            {"model_input_width", std::to_string(model_input_width)},
            {"model_input_height", std::to_string(model_input_height)},
            {"model_input_scale_mode", model_input_scale_mode},
            {"image_database_model_path", image_database_model_path.string()},
            {"similarity_threshold", std::to_string(similarity_threshold)}
            });
    LOG(INFO) << "Calculator graph: " << calculator_graph_str;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_str);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    std::string file_stem = video_path.stem().string();

    MP_RETURN_IF_ERROR(graph.ObserveOutputStream("output_stream", [&file_stem, &output_dir](const mediapipe::Packet& packet) {
                const mediapipe::Timestamp& timestamp = packet.Timestamp();
                const std::chrono::microseconds timestamp_us = std::chrono::microseconds(timestamp.Microseconds());

                const std::tuple<std::vector<std::pair<std::string, double>>, std::vector<float>, std::shared_ptr<const mediapipe::ImageFrame>> &label_score_embedding_image = packet.Get<std::tuple<std::vector<std::pair<std::string, double>>, std::vector<float>, std::shared_ptr<const mediapipe::ImageFrame>>>();
                std::vector<std::pair<std::string, double>> label_score_list = std::get<0>(label_score_embedding_image);

                std::vector<float> embedding = std::get<1>(label_score_embedding_image);

                std::string label;
                if (label_score_list.empty()) {
                    LOG(INFO) << "No label found";
                    return absl::OkStatus();
                } else if (label_score_list.size() == 1) {
                    label = label_score_list[0].first + "_" + score_to_label(label_score_list[0].second);
                } else {
                    std::vector<std::pair<std::string, double>> ordered_label_score_list = label_score_list;
                    std::sort(ordered_label_score_list.begin(), ordered_label_score_list.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                        return a.second > b.second;
                    });
                    std::vector<std::string> label_components;

                    for (const auto& label_score : ordered_label_score_list) {
                        const std::string& label = label_score.first;
                        label_components.push_back(label);

                        const double score = label_score.second;
                        LOG(INFO) << "Label: " << label_score.first << ", Score: " << label_score.second;

                        // score to 0 - 9 labels
                        label_components.push_back(score_to_label(score));
                    }

                    label = std::accumulate(std::next(label_components.begin()), label_components.end(), label_components[0], [](const std::string& a, const std::string& b) {
                        return a + "_" + b;
                    });
                }

                std::shared_ptr<const mediapipe::ImageFrame> frame = std::get<2>(label_score_embedding_image);
                std::unique_ptr<cv::Mat> frame_mat(new cv::Mat(frame->Height(), frame->Width(), CV_8UC3));
                if (frame_mat->channels() == 3) {
                    cv::cvtColor(mediapipe::formats::MatView(&*frame), *frame_mat, cv::COLOR_RGB2BGR);
                } else {
                    return absl::InternalError("Unsupported number of channels");
                }
                std::string timestamp_str = std::to_string(timestamp_us.count());
                int pad_len = std::max(0, 16 - static_cast<int>(timestamp_str.size()));
                std::string padded_timestamp_str = std::string(pad_len, '0') + timestamp_str;
                std::filesystem::path label_dir = output_dir / label;
                if (!std::filesystem::exists(label_dir)) {
                    std::filesystem::create_directories(label_dir);
                }
                std::filesystem::path output_path = output_dir / label / (file_stem + "_" + padded_timestamp_str + "_" + label + ".png");

                cv::imwrite(output_path.string(), *frame_mat);

                ImageDatabase &image_database = ImageDatabase::GetInstance();
                absl::Status insertion_status = image_database.Insert(output_path, embedding);
                if (!insertion_status.ok()) {
                    LOG(ERROR) << "Insert Embedding Error: " << insertion_status.message();
                    return insertion_status;
                }
                
                return absl::OkStatus();
                }));

    MP_RETURN_IF_ERROR(graph.StartRun({
                { "input_video_path", mediapipe::MakePacket<std::string>(video_path.string()) }
                }));

    MP_RETURN_IF_ERROR(graph.WaitUntilDone());

    return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<sqlite3, decltype(&sqlite3_close)>> OpenDatabaseAndMigrateIfNeeded() {
    std::filesystem::path db_path = "./main.db";
    sqlite3* db_ptr;
    int r = sqlite3_open(db_path.string().c_str(), &db_ptr);
    RET_CHECK(r == SQLITE_OK);
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db(db_ptr, sqlite3_close);

    // check version table
    const char* version_table_sql = "CREATE TABLE IF NOT EXISTS version (version INTEGER)";
    char* err_msg = nullptr;
    r = sqlite3_exec(db.get(), version_table_sql, nullptr, nullptr, &err_msg);
    RET_CHECK(r == SQLITE_OK);

    {
        int version = 1;

        const char* version_query = "SELECT version FROM version WHERE version = ?";
        sqlite3_stmt* stmt_ptr;
        r = sqlite3_prepare_v2(db.get(), version_query, -1, &stmt_ptr, nullptr);
        RET_CHECK(r == SQLITE_OK);
        std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt(stmt_ptr, sqlite3_finalize);

        r = sqlite3_bind_int(stmt.get(), 1, version);
        RET_CHECK(r == SQLITE_OK);

        r = sqlite3_step(stmt.get());
        RET_CHECK(r == SQLITE_DONE || r == SQLITE_ROW);
        if (r == SQLITE_DONE) {
            LOG(INFO) << "Empty version table, migrating";

            const char* create_video_table_sql = "CREATE TABLE video (id INTEGER PRIMARY KEY, path TEXT UNIQUE, done BOOLEAN)";
            r = sqlite3_exec(db.get(), create_video_table_sql, nullptr, nullptr, &err_msg);
            RET_CHECK(r == SQLITE_OK);

            const char* insert_version_sql = "INSERT INTO version (version) VALUES (?)";
            sqlite3_stmt* insert_stmt_ptr;
            r = sqlite3_prepare_v2(db.get(), insert_version_sql, -1, &insert_stmt_ptr, nullptr);
            RET_CHECK(r == SQLITE_OK);
            std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> insert_stmt(insert_stmt_ptr, sqlite3_finalize);
            r = sqlite3_bind_int(insert_stmt.get(), 1, version);
            RET_CHECK(r == SQLITE_OK);
            r = sqlite3_step(insert_stmt.get());
            RET_CHECK(r == SQLITE_DONE);
        }
    }

    return std::move(db);
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    // Initialize ImageDatabase singleton
    std::filesystem::path image_database_model_path = "mobilenet_v3_large.tflite";
    ImageDatabase::Initialize(image_database_model_path, 1280);
    ImageDatabase &image_database = ImageDatabase::GetInstance();

    absl::ParseCommandLine(argc, argv);
    std::string video_dir_str = absl::GetFlag(FLAGS_video_dir);
    if (video_dir_str.empty()) {
        LOG(ERROR) << "Please specify video dir with --video_dir flag";
        return EXIT_FAILURE;
    }
    std::filesystem::path video_dir(video_dir_str);
    if (!std::filesystem::exists(video_dir)) {
        LOG(ERROR) << "Video dir does not exist";
        return EXIT_FAILURE;
    }


    std::string model_path_str = absl::GetFlag(FLAGS_model_path);
    if (model_path_str.empty()) {
        LOG(ERROR) << "Please specify model path with --model_path flag";
        return EXIT_FAILURE;
    }
    std::filesystem::path model_path(model_path_str);
    if (!std::filesystem::exists(model_path)) {
        LOG(ERROR) << "Model path does not exist";
        return EXIT_FAILURE;
    }

    std::string output_dir_str = absl::GetFlag(FLAGS_output_dir);
    if (output_dir_str.empty()) {
        LOG(ERROR) << "Please specify output dir with --output_dir flag";
        return EXIT_FAILURE;
    }
    std::filesystem::path output_dir(output_dir_str);
    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    double similarity_threshold = absl::GetFlag(FLAGS_similarity_threshold);

    uint model_input_width = absl::GetFlag(FLAGS_model_input_width);
    uint model_input_height = absl::GetFlag(FLAGS_model_input_height);
    std::pair<uint, uint> model_input_size(model_input_width, model_input_height);
    std::string model_input_scale_mode = absl::GetFlag(FLAGS_model_input_scale_mode);
    double capture_period = absl::GetFlag(FLAGS_capture_period);

    absl::StatusOr<std::unique_ptr<sqlite3, decltype(&sqlite3_close)>> status_or_db = OpenDatabaseAndMigrateIfNeeded();
    if (!status_or_db.ok()) {
        LOG(ERROR) << "Error: " << status_or_db.status().message();
        return EXIT_FAILURE;
    }
    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db = std::move(status_or_db.value());

    for (const auto& entry : std::filesystem::directory_iterator(video_dir)) {
        std::filesystem::path video_path = entry.path();
        LOG(INFO) << "Processing video: " << video_path;

        std::string extension = video_path.extension().string();
        if (extension != ".mp4" && extension != ".avi" && extension != ".mov" && extension != ".mkv" && extension != ".flv" && extension != ".webm" && extension != ".wmv") {
            LOG(INFO) << "Skipping video: " << video_path;
            continue;
        }

        const char* select_video_sql = "SELECT done FROM video WHERE path = ?";
        sqlite3_stmt* select_stmt_ptr;
        int r = sqlite3_prepare_v2(db.get(), select_video_sql, -1, &select_stmt_ptr, nullptr);
        if (r != SQLITE_OK) {
            LOG(ERROR) << "Error: " << sqlite3_errmsg(db.get());
            return EXIT_FAILURE;
        }
        std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> select_stmt(select_stmt_ptr, sqlite3_finalize);
        r = sqlite3_bind_text(select_stmt.get(), 1, video_path.c_str(), -1, SQLITE_STATIC);
        if (r != SQLITE_OK) {
            LOG(ERROR) << "Error: " << sqlite3_errmsg(db.get());
            return EXIT_FAILURE;
        }
        r = sqlite3_step(select_stmt.get());
        if (r != SQLITE_DONE && r != SQLITE_ROW) {
            LOG(ERROR) << "Error: " << sqlite3_errmsg(db.get());
            return EXIT_FAILURE;
        }
        if (r == SQLITE_ROW) {
            int done = sqlite3_column_int(select_stmt.get(), 0);
            if (done) {
                LOG(INFO) << "Video already processed: " << video_path;
                continue;
            }
        }

        absl::Status mediapipe_status = RunMediapipe(video_path, model_path, output_dir, capture_period, model_input_size, model_input_scale_mode, similarity_threshold);
        if (!mediapipe_status.ok()) {
            LOG(WARNING) << "Mediapipe Process Failed, skip and next: " << mediapipe_status.message();
        }

        const char* insert_video_sql = "INSERT INTO video (path, done) VALUES (?, ?)";
        sqlite3_stmt* insert_stmt_ptr;
        r = sqlite3_prepare_v2(db.get(), insert_video_sql, -1, &insert_stmt_ptr, nullptr);
        if (r != SQLITE_OK) {
            LOG(ERROR) << "Error: " << sqlite3_errmsg(db.get());
            return EXIT_FAILURE;
        }

        LOG(INFO) << "Marking video as done: " << video_path;

        std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> insert_stmt(insert_stmt_ptr, sqlite3_finalize);
        r = sqlite3_bind_text(insert_stmt.get(), 1, video_path.c_str(), -1, SQLITE_STATIC);
        if (r != SQLITE_OK) {
            LOG(ERROR) << "Error: " << sqlite3_errmsg(db.get());
            return EXIT_FAILURE;
        }
        r = sqlite3_bind_int(insert_stmt.get(), 2, 1);
        if (r != SQLITE_OK) {
            LOG(ERROR) << "Error: " << sqlite3_errmsg(db.get());
            return EXIT_FAILURE;
        }
        r = sqlite3_step(insert_stmt.get());
        if (r != SQLITE_DONE) {
            LOG(ERROR) << "Error: " << sqlite3_errmsg(db.get());
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}


