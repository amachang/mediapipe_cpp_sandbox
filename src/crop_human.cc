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

        # Detect pose
        node {
            calculator: "PoseDetectionCpu"
            input_stream: "IMAGE:input_stream"
            output_stream: "DETECTIONS:pose_detections"
        }
        # in my understanding, this detection is originally a single detection
        node {
            calculator: "SplitDetectionVectorCalculator"
            input_stream: "pose_detections"
            output_stream: "pose_detection"
            node_options: {
                [type.googleapis.com/mediapipe.SplitVectorCalculatorOptions] {
                    ranges: { begin: 0 end: 1 }
                    element_only: true
                }
            }
        }

        # Image size
        node {
            calculator: "ImagePropertiesCalculator"
            input_stream: "IMAGE_CPU:input_stream"
            output_stream: "SIZE:input_image_size"
        }

        # Smoothing
        node {
            calculator: "DetectionToLandmarksCalculator"
            input_stream: "DETECTION:pose_detection"
            output_stream: "LANDMARKS:pose_landmarks"
        }
        node {
            calculator: "LandmarksSmoothingCalculator"
            input_stream: "NORM_LANDMARKS:pose_landmarks"
            input_stream: "IMAGE_SIZE:input_image_size"
            output_stream: "NORM_FILTERED_LANDMARKS:smooth_pose_landmarks"
            node_options: {
                [type.googleapis.com/mediapipe.LandmarksSmoothingCalculatorOptions] {
                    velocity_filter: {
                        window_size: 10
                        velocity_scale: 0.001
                        disable_value_scaling: true
                    }
                }
            }
        }
        node {
            calculator: "LandmarksToDetectionCalculator"
            input_stream: "NORM_LANDMARKS:smooth_pose_landmarks"
            output_stream: "DETECTION:smooth_pose_detection"
        }

        # Calculate rect
        node {
            calculator: "DetectionsToRectsCalculator"
            input_stream: "DETECTION:smooth_pose_detection"
            input_stream: "IMAGE_SIZE:input_image_size"
            output_stream: "NORM_RECT:smooth_pose_detection_rect"
            node_options: {
                [type.googleapis.com/mediapipe.DetectionsToRectsCalculatorOptions] {
                    conversion_mode: USE_BOUNDING_BOX
                }
            }
        }
        node {
            calculator: "RectTransformationCalculator"
            input_stream: "NORM_RECT:smooth_pose_detection_rect"
            input_stream: "IMAGE_SIZE:input_image_size"
            output_stream: "smooth_pose_rect"
            node_options: {
                [type.googleapis.com/mediapipe.RectTransformationCalculatorOptions] {
                    square_long: true
                }
            }
        }

        # Crop image
        node {
            calculator: "ImageCroppingCalculator"
            input_stream: "IMAGE:input_stream"
            input_stream: "NORM_RECT:smooth_pose_rect"
            output_stream: "IMAGE:output_stream"
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

