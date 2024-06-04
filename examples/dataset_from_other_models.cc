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
ABSL_FLAG(double, score_threshold, 0.8, "score threshold");

absl::Status RunMediapipe(
        const std::filesystem::path& video_path, 
        const std::filesystem::path& model_path,
        const std::filesystem::path& output_dir,
        const double capture_period,
        const std::pair<uint, uint> model_input_size,
        const std::string& model_input_scale_mode,
        double score_threshold
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

        # Classify image
        node {
            calculator: "ToImageCalculator"
            input_stream: "IMAGE:square_input_image"
            output_stream: "IMAGE:used_by_classifier_image"
        }
        node {
            calculator: "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph"
            input_stream: "IMAGE:used_by_classifier_image"
            output_stream: "CLASSIFICATIONS:classifications"
            output_stream: "IMAGE:classified_image"
            node_options: {
                [type.googleapis.com/mediapipe.tasks.vision.image_classifier.proto.ImageClassifierGraphOptions] {
                    base_options: {
                        model_asset: {
                            file_name: "{{ model_path }}" # TODO escape path
                        }
                    }
                    classifier_options: {
                        max_results: 1
                        score_threshold: {{ score_threshold }}
                    }
                }
            }
        }

        # To callback
        node {
            calculator: "ToImageCalculator"
            input_stream: "IMAGE:cropped_image_stream"
            output_stream: "IMAGE:type_converted_image"
        }
        node {
            calculator: "ClassificationImageGateCalculator"
            input_stream: "CLASSIFICATIONS:classifications"
            input_stream: "IMAGE:type_converted_image"
            output_stream: "LABEL_SCORE_IMAGE:output_stream"
        }
    )pb");

    std::string calculator_graph_str = env.render(calculator_graph_template, {
            {"model_path", model_path.string()},
            {"capture_period_us", std::to_string(capture_period_us)},
            {"model_input_width", std::to_string(model_input_width)},
            {"model_input_height", std::to_string(model_input_height)},
            {"model_input_scale_mode", model_input_scale_mode},
            {"score_threshold", std::to_string(score_threshold)}
            });
    LOG(INFO) << "Calculator graph: " << calculator_graph_str;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_str);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    std::string file_stem = video_path.stem().string();

    MP_RETURN_IF_ERROR(graph.ObserveOutputStream("output_stream", [&file_stem, &output_dir](const mediapipe::Packet& packet) {
                const mediapipe::Timestamp& timestamp = packet.Timestamp();
                const std::chrono::microseconds timestamp_us = std::chrono::microseconds(timestamp.Microseconds());
                LOG(INFO) << "Received frame at " << timestamp.Microseconds();

                std::pair<std::vector<std::pair<std::string, double>>, std::shared_ptr<const mediapipe::ImageFrame>> label_score_frame = packet.Get<std::pair<std::vector<std::pair<std::string, double>>, std::shared_ptr<const mediapipe::ImageFrame>>>();
                std::vector<std::pair<std::string, double>> label_score_list = label_score_frame.first;
                if (label_score_list.empty()) {
                    LOG(INFO) << "No label found";
                    return absl::OkStatus();
                }
                std::pair<std::string, double> label_score = label_score_list[0];
                std::string label = label_score.first;

                std::shared_ptr<const mediapipe::ImageFrame> frame = label_score_frame.second;
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

    uint model_input_width = absl::GetFlag(FLAGS_model_input_width);
    uint model_input_height = absl::GetFlag(FLAGS_model_input_height);
    std::pair<uint, uint> model_input_size(model_input_width, model_input_height);
    std::string model_input_scale_mode = absl::GetFlag(FLAGS_model_input_scale_mode);
    double capture_period = absl::GetFlag(FLAGS_capture_period);
    double score_threshold = absl::GetFlag(FLAGS_score_threshold);

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

        absl::Status mediapipe_status = RunMediapipe(video_path, model_path, output_dir, capture_period, model_input_size, model_input_scale_mode, score_threshold);
        if (!mediapipe_status.ok()) {
            LOG(ERROR) << "Error: " << mediapipe_status.message();
            return EXIT_FAILURE;
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


