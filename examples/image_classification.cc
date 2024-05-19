#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <filesystem>
#include <thread>
#include <mutex>
#include <utility>

#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "glog/logging.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"

ABSL_FLAG(std::string, video_dir, "", "video dir");
ABSL_FLAG(std::string, dataset_dir, "train_workspace", "dataset dir");

std::sig_atomic_t g_inturrpted_signal_received = false;
void interrupted_signal_handler(int signal) {
    LOG(INFO) << "Received interrupted signal: " << signal;
    g_inturrpted_signal_received = signal;
}

std::unordered_set<std::string> g_video_extensions = {
    ".mp4", ".avi", ".mov", ".mkv"
};

absl::Status RunGraph(const std::filesystem::path& input_video_path, const std::filesystem::path& dataset_dir) {
    std::string input_video_filename = input_video_path.stem().string();

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        output_stream: "output_stream"

        # Input video
        node {
            calculator: "OpenCvVideoDecoderCalculator"
            input_side_packet: "INPUT_FILE_PATH:input_video_path"
            output_stream: "VIDEO:input_stream"
            output_stream: "VIDEO_PRESTREAM:input_video_header"
        }

        node {
            calculator: "PacketThinnerCalculator"
            input_stream: "input_stream"
            output_stream: "downsampled_input_image"
            node_options: {
                [type.googleapis.com/mediapipe.PacketThinnerCalculatorOptions] {
                    thinner_type: ASYNC
                    period: 10000000
                }
            }
        }

        # ImageFrame -> Image (just type conversion)
        node {
            calculator: "ToImageCalculator"
            input_stream: "IMAGE:downsampled_input_image"
            output_stream: "IMAGE:used_by_classifier_image"
        }

        node {
            calculator: "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph"
            input_stream: "IMAGE:used_by_classifier_image"
            output_stream: "CLASSIFICATIONS:classifications"
            output_stream: "IMAGE:classified_image"
            node_options {
                [type.googleapis.com/mediapipe.tasks.vision.image_classifier.proto.ImageClassifierGraphOptions] {
                    base_options {
                        model_asset {
                            file_name: "models/efficientnet_lite2.tflite"
                        }
                    }
                    classifier_options {
                        max_results: 1
                        score_threshold: 0.2
                    }
                }
            }
        }

        node {
            calculator: "ClassificationImageGateCalculator"
            input_stream: "CLASSIFICATIONS:classifications"
            input_stream: "IMAGE:classified_image"
            output_stream: "LABEL_AND_IMAGE:output_stream"
        }
    )pb");


    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    std::shared_ptr<std::mutex> status_mutex = std::make_shared<std::mutex>();
    std::shared_ptr<absl::Status> error_status = std::make_shared<absl::Status>(absl::OkStatus());

    std::shared_ptr<std::mutex> mkdir_mutex = std::make_shared<std::mutex>();
    std::shared_ptr<std::chrono::steady_clock::time_point> last_packet_received_time = std::make_shared<std::chrono::steady_clock::time_point>(std::chrono::steady_clock::now());

    {
        std::weak_ptr<std::mutex> status_mutex_weak = status_mutex;
        std::weak_ptr<std::mutex> mkdir_mutex_weak = mkdir_mutex;
        std::weak_ptr<absl::Status> error_status_weak = error_status;
        std::weak_ptr<std::chrono::steady_clock::time_point> last_packet_received_time_weak = last_packet_received_time;

        MP_RETURN_IF_ERROR(graph.ObserveOutputStream("output_stream", [
                    status_mutex_weak = std::move(status_mutex_weak),
                    mkdir_mutex_weak = std::move(mkdir_mutex_weak),
                    error_status_weak = std::move(error_status_weak),
                    last_packet_received_time_weak = std::move(last_packet_received_time_weak),
                    dataset_dir = std::move(dataset_dir),
                    inpunt_video_filename = std::move(input_video_filename)](const mediapipe::Packet& packet) {
            std::shared_ptr<std::mutex> status_mutex = status_mutex_weak.lock();
            std::shared_ptr<std::mutex> mkdir_mutex = mkdir_mutex_weak.lock();
            std::shared_ptr<absl::Status> error_status = error_status_weak.lock();
            std::shared_ptr<std::chrono::steady_clock::time_point> last_packet_received_time = last_packet_received_time_weak.lock();
            if (!status_mutex || !error_status || !mkdir_mutex || !last_packet_received_time) {
                LOG(ERROR) << "Callback called after the object was destroyed";
                return absl::OkStatus();
            }

            {
                std::lock_guard<std::mutex> lock(*status_mutex);
                *last_packet_received_time = std::chrono::steady_clock::now();
            }

            std::pair<std::string, std::shared_ptr<const mediapipe::ImageFrame>> label_and_frame = packet.Get<std::pair<std::string, std::shared_ptr<const mediapipe::ImageFrame>>>();
            const std::string& label = label_and_frame.first;
            const std::shared_ptr<const mediapipe::ImageFrame> output_frame = label_and_frame.second;
            mediapipe::Timestamp timestamp = packet.Timestamp();

            cv::Mat output_frame_mat_bgr;
            try {
                cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
                cv::cvtColor(output_frame_mat, output_frame_mat_bgr, cv::COLOR_RGB2BGR);
            } catch (const cv::Exception& e) {
                LOG(ERROR) << "Failed to convert ImageFrame to cv::Mat: " << e.what();
                error_status->Update(absl::InternalError("Failed to convert ImageFrame to cv::Mat"));
                return absl::OkStatus();
            }

            std::filesystem::path label_dir = dataset_dir / label;

            // double check for performance
            if (!std::filesystem::exists(label_dir)) {
                std::lock_guard<std::mutex> lock(*mkdir_mutex);
                // make dir if not exists
                if (!std::filesystem::exists(label_dir)) {
                    std::filesystem::create_directories(label_dir);
                }
            }

            std::filesystem::path output_image_path = label_dir / (inpunt_video_filename + "_" + std::to_string(timestamp.Value()) + ".png");

            try {
                cv::imwrite(output_image_path.string(), output_frame_mat_bgr);
            } catch (const cv::Exception& e) {
                LOG(ERROR) << "Failed to write image to file: " << e.what();
                error_status->Update(absl::InternalError("Failed to write image to file"));
                return absl::OkStatus();
            }
            std::cout << "Label: " << label << std::endl;

            return absl::OkStatus();
        }));
    }

    {
        std::weak_ptr<std::mutex> status_mutex_weak = status_mutex;
        std::weak_ptr<absl::Status> error_status_weak = error_status;

        graph.SetErrorCallback([status_mutex_weak = std::move(status_mutex_weak), error_status_weak = std::move(error_status_weak)](absl::Status status) {
            LOG(ERROR) << "Error: " << status.message();

            std::shared_ptr<std::mutex> status_mutex = status_mutex_weak.lock();
            std::shared_ptr<absl::Status> error_status = error_status_weak.lock();
            if (!status_mutex || !error_status) {
                LOG(ERROR) << "Error callback called after the object was destroyed";
            }
            {
                std::lock_guard<std::mutex> lock(*status_mutex);
                error_status->Update(status);
            }
        });
    }

    MP_RETURN_IF_ERROR(graph.StartRun({
                { "input_video_path", mediapipe::MakePacket<std::string>(input_video_path) }
                }));

    while (!g_inturrpted_signal_received) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        {
            std::lock_guard<std::mutex> lock(*status_mutex);
            if (!error_status->ok()) {
                break;
            }

            std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = now - *last_packet_received_time;
            if (elapsed_seconds.count() > 10) {
                LOG(ERROR) << "No packet received for 10 seconds. Exiting...";
                break;
            }
        }
    }

    absl::Status close_status = graph.CloseAllPacketSources();
    absl::Status wait_until_done_status = graph.WaitUntilDone();

    {
        std::lock_guard<std::mutex> lock(*status_mutex);
        MP_RETURN_IF_ERROR(*error_status);
    }
    MP_RETURN_IF_ERROR(close_status);
    MP_RETURN_IF_ERROR(wait_until_done_status);

    return absl::OkStatus();
}

int main(int argc, char** argv) {
    std::signal(SIGINT, interrupted_signal_handler);
    google::InitGoogleLogging(argv[0]);

    absl::ParseCommandLine(argc, argv);
    std::string video_dir_str = absl::GetFlag(FLAGS_video_dir);
    if (video_dir_str.empty()) {
        LOG(ERROR) << "You must specify a video path using --video_dir";
        return EXIT_FAILURE;
    }

    std::string dataset_dir_str = absl::GetFlag(FLAGS_dataset_dir);
    if (dataset_dir_str.empty()) {
        LOG(ERROR) << "You must specify a dataset path using --dataset_dir";
        return EXIT_FAILURE;
    }

    std::filesystem::path video_dir(video_dir_str);
    const std::filesystem::path dataset_dir(dataset_dir_str);

    unsigned int consecutive_errors = 0;

    // readdir
    for (const auto& entry : std::filesystem::directory_iterator(video_dir)) {
        const std::filesystem::path& video_path = entry.path();
        if (entry.is_directory()) {
            LOG(INFO) << "Skipping directory: " << video_path.string();
            continue;
        }

        std::string ext = video_path.extension().string();
        if (g_video_extensions.find(ext) == g_video_extensions.end()) {
            LOG(INFO) << "Skipping non-video file: " << video_path.string();
            continue;
        }

        LOG(INFO) << "Processing video: " << video_path.string();
        absl::Status status = RunGraph(video_path, dataset_dir);
        if (!status.ok()) {
            LOG(ERROR) << "Failed to run the graph: " << status.message();
            consecutive_errors++;
            if (consecutive_errors >= 3) {
                LOG(ERROR) << "Too many consecutive errors. Exiting...";
                break;
            }
        } else {
            consecutive_errors = 0;
        }

        if (g_inturrpted_signal_received) {
            break;
        }
    }

    LOG(INFO) << "Graph run successfully";
    return EXIT_SUCCESS;
}


