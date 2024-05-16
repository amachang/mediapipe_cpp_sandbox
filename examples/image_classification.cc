#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

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

ABSL_FLAG(std::string, input_video_path, "", "path of video to load");

std::sig_atomic_t g_inturrpted_signal_received = false;
void interrupted_signal_handler(int signal) {
    LOG(INFO) << "Received interrupted signal: " << signal;
    g_inturrpted_signal_received = signal;
}

absl::Status RunGraph(std::string&& input_video_path) {
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
                    period: 100000
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
            output_stream: "IMAGE:output_stream"
            node_options {
                [type.googleapis.com/mediapipe.tasks.vision.image_classifier.proto.ImageClassifierGraphOptions] {
                    base_options {
                        model_asset {
                            file_name: "models/efficientnet_lite2.tflite"
                        }
                    }
                    classifier_options {
                        max_results: 3
                        score_threshold: 0.6
                    }
                }
            }
        }
    )pb");

    std::shared_ptr<std::mutex> video_frames_mutex = std::make_shared<std::mutex>();
    std::shared_ptr<std::deque<cv::Mat>> video_frames = std::make_shared<std::deque<cv::Mat>>();

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    std::weak_ptr<std::mutex> video_frames_mutex_weak = video_frames_mutex;
    std::weak_ptr<std::deque<cv::Mat>> video_frames_weak = video_frames;
    MP_RETURN_IF_ERROR(graph.ObserveOutputStream("classifications", [](const mediapipe::Packet& packet) {
        const mediapipe::tasks::components::containers::proto::ClassificationResult& classification_result = packet.Get<mediapipe::tasks::components::containers::proto::ClassificationResult>();
        for (const auto& classifications : classification_result.classifications()) {
            const auto num_classifications = classifications.classification_list().classification_size();
            if (num_classifications == 0) {
                continue;
            }
            std::cout << "Detection Count: " << num_classifications << std::endl;
            for (const auto& classification : classifications.classification_list().classification()) {
                std::cout << "Class: " << classification.label() << ", Score: " << classification.score() << std::endl;
            }
        }

        return absl::OkStatus();
    }));
    MP_RETURN_IF_ERROR(graph.ObserveOutputStream("output_stream", [video_frames_weak = std::move(video_frames_weak), video_frames_mutex_weak = std::move(video_frames_mutex_weak)](const mediapipe::Packet& packet) {
        std::shared_ptr<std::deque<cv::Mat>> video_frames = video_frames_weak.lock();
        std::shared_ptr<std::mutex> video_frames_mutex = video_frames_mutex_weak.lock();
        if (!video_frames || !video_frames_mutex) {
            LOG(WARNING) << "Video frames buffer is already released";
            return absl::OkStatus();
        }

        const mediapipe::Image& output_image = packet.Get<mediapipe::Image>();
        const std::shared_ptr<mediapipe::ImageFrame> output_frame = output_image.GetImageFrameSharedPtr();
        cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
        cv::Mat output_frame_mat_bgr;
        cv::cvtColor(output_frame_mat, output_frame_mat_bgr, cv::COLOR_RGB2BGR);

        {
            std::lock_guard<std::mutex> lock(*video_frames_mutex);
            video_frames->push_back(output_frame_mat_bgr);
        }

        return absl::OkStatus();
    }));

    LOG(INFO) << "Start running the graph";
    MP_RETURN_IF_ERROR(graph.StartRun({
                { "input_video_path", mediapipe::MakePacket<std::string>(input_video_path) },
                }));

    cv::namedWindow("Output", /*flags=WINDOW_AUTOSIZE*/ 1);
    while (!g_inturrpted_signal_received) {
        if (graph.HasError()) {
            absl::Status status;
            RET_CHECK(graph.GetCombinedErrors(&status));
            return status;
        }
        {
            std::lock_guard<std::mutex> lock(*video_frames_mutex);
            if (!video_frames->empty()) {
                LOG(INFO) << "Displaying frame";
                cv::imshow("Output", video_frames->front());
                video_frames->pop_front();
            }
        }
        const int pressed_key = cv::waitKey(1);
        if (pressed_key >= 0 && pressed_key != 255) {
            break;
        }
    }

    LOG(INFO) << "Shutting down graph";
    MP_RETURN_IF_ERROR(graph.CloseAllPacketSources());
    MP_RETURN_IF_ERROR(graph.WaitUntilDone());

    return absl::OkStatus();
}

int main(int argc, char** argv) {
    std::signal(SIGINT, interrupted_signal_handler);
    google::InitGoogleLogging(argv[0]);

    absl::ParseCommandLine(argc, argv);
    std::string input_video_path = absl::GetFlag(FLAGS_input_video_path);
    if (input_video_path.empty()) {
        LOG(ERROR) << "You must specify a video path using --input_video_path";
        return EXIT_FAILURE;
    }

    absl::Status status = RunGraph(std::move(input_video_path));
    if (!status.ok()) {
        LOG(ERROR) << "Failed to run the graph: " << status.message();
        return EXIT_FAILURE;
    }

    LOG(INFO) << "Graph run successfully";
    return EXIT_SUCCESS;
}


