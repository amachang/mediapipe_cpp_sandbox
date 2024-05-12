#include <cstdlib>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

static_assert((CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2), "Only opencv 3.2+ is supported.");

constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, input_video_path, "", "Full path of video to load. If not provided, attempt to use a webcam.");

absl::Status RunMPPGraph() {
    LOG(INFO) << "Start running MediaPipe graph.";
    std::string video_path = absl::GetFlag(FLAGS_input_video_path);
    RET_CHECK(!video_path.empty());

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        input_stream: "in"
        output_stream: "out"
        output_stream: "multi_face_landmarks"

        # define side packets
        node {
            calculator: "ConstantSidePacketCalculator"
            output_side_packet: "PACKET:0:num_faces"
            output_side_packet: "PACKET:1:with_attention"
            node_options: {
                [type.googleapis.com/mediapipe.ConstantSidePacketCalculatorOptions]: {
                    packet { int_value: 1 }
                    packet { bool_value: true }
                }
            }
        }
        
        node {
            calculator: "FaceLandmarkFrontCpu"
            input_stream: "IMAGE:in"
            input_side_packet: "NUM_FACES:num_faces"
            input_side_packet: "WITH_ATTENTION:with_attention"
            output_stream: "LANDMARKS:multi_face_landmarks"
            output_stream: "ROIS_FROM_LANDMARKS:face_rects_from_landmarks"
            output_stream: "DETECTIONS:face_detections"
            output_stream: "ROIS_FROM_DETECTIONS:face_rects_from_detections"
        }
        
        node {
            calculator: "FaceRendererCpu"
            input_stream: "IMAGE:in"
            input_stream: "LANDMARKS:multi_face_landmarks"
            input_stream: "NORM_RECTS:face_rects_from_landmarks"
            input_stream: "DETECTIONS:face_detections"
            output_stream: "IMAGE:out"
        }
      )pb");

    LOG(INFO) << "Start creating MediaPipe graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    LOG(INFO) << "Making video stream from: " << video_path;
    cv::VideoCapture capture;
    capture.open(std::move(video_path));
    RET_CHECK(capture.isOpened());
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);

    LOG(INFO) << "Making video writer.";
    cv::VideoWriter writer;
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);

    LOG(INFO) << "Start running the MediaPipe graph.";
    MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller("out"));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    LOG(INFO) << "Start grabbing and processing frames.";
    while (true) {
        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty()) {
            LOG(INFO) << "Empty frame, end of video reached.";
            break;
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        // Send image packet into the graph.
        size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream("in", mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        // Get the graph result packet, or stop if that fails.
        mediapipe::Packet packet;
        if (!poller.Next(&packet)) {
            break;
        }
        const mediapipe::ImageFrame& output_frame = packet.Get<mediapipe::ImageFrame>();

        // Convert back to opencv for display or saving.
        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        // Display the output frame.
        cv::imshow(kWindowName, output_frame_mat);

        // Press any key to exit.
        const int pressed_key = cv::waitKey(5);
        if (pressed_key >= 0 && pressed_key != 255) {
            break;
        }
    }

    if (writer.isOpened()) {
        writer.release();
    }
    MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));

    return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    } else {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}

