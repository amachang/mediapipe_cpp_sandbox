#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "glog/logging.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"

ABSL_FLAG(std::string, input_video_path, "", "path of video to load");

std::sig_atomic_t g_inturrpted_signal_received = false;
void interrupted_signal_handler(int signal) {
    LOG(INFO) << "Received interrupted signal: " << signal;
    g_inturrpted_signal_received = signal;
}

absl::Status RunGraph(std::string&& input_video_path) {
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
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
            calculator: "AlignmentPointsRectsCalculator"
            input_stream: "DETECTION:smooth_pose_detection"
            input_stream: "IMAGE_SIZE:input_image_size"
            output_stream: "NORM_RECT:smooth_pose_rect"
            node_options: {
                [type.googleapis.com/mediapipe.DetectionsToRectsCalculatorOptions] {
                    rotation_vector_start_keypoint_index: 0
                    rotation_vector_end_keypoint_index: 1
                    rotation_vector_target_angle_degrees: 90
                }
            }
        }

        # Render pose detections
        node {
            calculator: "RectToRenderDataCalculator"
            input_stream: "NORM_RECT:smooth_pose_rect"
            output_stream: "RENDER_DATA:smooth_pose_render_data"
            node_options: {
                [type.googleapis.com/mediapipe.RectToRenderDataCalculatorOptions] {
                    thickness: 1.0
                    color { r: 255 g: 0 b: 0 }
                }
            }
        }
        node {
            calculator: "AnnotationOverlayCalculator"
            input_stream: "IMAGE:input_stream"
            input_stream: "smooth_pose_render_data"
            output_stream: "IMAGE:annotated_image_stream"
        }

        # Output video
        node {
            calculator: "OpenCvVideoEncoderCalculator"
            input_side_packet: "OUTPUT_FILE_PATH:output_video_path"
            input_stream: "VIDEO:annotated_image_stream"
            input_stream: "VIDEO_PRESTREAM:input_video_header"
            node_options: {
                [type.googleapis.com/mediapipe.OpenCvVideoEncoderCalculatorOptions]: {
                    codec: "avc1"
                    video_format: "mp4"
                }
            }
        }
    )pb");

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    LOG(INFO) << "Start running the graph";
    MP_RETURN_IF_ERROR(graph.StartRun({
                { "input_video_path", mediapipe::MakePacket<std::string>(input_video_path) },
                { "output_video_path", mediapipe::MakePacket<std::string>("output.mp4") },
                }));

    while (!g_inturrpted_signal_received) {
        if (graph.HasError()) {
            absl::Status status;
            RET_CHECK(graph.GetCombinedErrors(&status));
            return status;
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

