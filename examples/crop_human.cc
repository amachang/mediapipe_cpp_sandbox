#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

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
            output_stream: "downsampled_input_stream"
            options: {
                [mediapipe.PacketThinnerCalculatorOptions.ext]: {
                    thinner_type: ASYNC
                    period: 100000
                }
            }
        }

        # Image size
        node {
            calculator: "ImagePropertiesCalculator"
            input_stream: "IMAGE_CPU:downsampled_input_stream"
            output_stream: "SIZE:input_image_size"
        }

        # Detect pose
        node {
            calculator: "PoseDetectionCpu"
            input_stream: "IMAGE:downsampled_input_stream"
            output_stream: "DETECTIONS:pose_detections"
        }

        # Detection to Rect
        node {
            calculator: "DetectionToLargestSquareRectCalculator"
            input_stream: "DETECTIONS:pose_detections"
            input_stream: "IMAGE_SIZE:input_image_size"
            output_stream: "pose_rect"
            node_options: {
                [type.googleapis.com/DetectionToLargestSquareRectCalculatorOptions] {
                    relative_margin: 0.2
                    min_score: 0.8
                }
            }
        }

        # Crop image
        node {
            calculator: "ImageCroppingCalculator"
            input_stream: "IMAGE:downsampled_input_stream"
            input_stream: "NORM_RECT:pose_rect"
            output_stream: "IMAGE:cropped_image_stream"
        }

        # Scale image
        node {
            calculator: "ScaleImageCalculator"
            input_stream: "cropped_image_stream"
            output_stream: "output_stream"
            node_options: {
                [type.googleapis.com/mediapipe.ScaleImageCalculatorOptions] {
                    target_width: 224
                    target_height: 224
                    output_format: SRGB
                    algorithm: DEFAULT
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
    MP_RETURN_IF_ERROR(graph.ObserveOutputStream("output_stream", [video_frames_weak = std::move(video_frames_weak), video_frames_mutex_weak = std::move(video_frames_mutex_weak)](const mediapipe::Packet& packet) {
        std::shared_ptr<std::deque<cv::Mat>> video_frames = video_frames_weak.lock();
        std::shared_ptr<std::mutex> video_frames_mutex = video_frames_mutex_weak.lock();
        if (!video_frames || !video_frames_mutex) {
            LOG(WARNING) << "Video frames buffer is already released";
            return absl::OkStatus();
        }

        const mediapipe::ImageFrame& output_frame = packet.Get<mediapipe::ImageFrame>();
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
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

