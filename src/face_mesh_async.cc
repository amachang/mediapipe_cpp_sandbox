#include <csignal>
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

volatile std::sig_atomic_t gSignalStatus;

static_assert((CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2), "Only opencv 3.2+ is supported.");

constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, input_video_path, "", "path of video to load");
ABSL_FLAG(std::string, output_video_path, "", "path of video to save (mp4 only)");

void signal_handler(int signal) {
    gSignalStatus = signal;
}

absl::Status RunMPPGraph() {
    LOG(INFO) << "Start running MediaPipe graph.";

    std::signal(SIGINT, signal_handler);

    std::string video_path = absl::GetFlag(FLAGS_input_video_path);
    RET_CHECK(!video_path.empty());

    std::string output_path = absl::GetFlag(FLAGS_output_video_path);
    if (output_path.empty()) {
        output_path = "output.mp4";
    }

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

    std::shared_ptr<std::mutex> video_writer_mutex = std::make_shared<std::mutex>();
    std::shared_ptr<cv::VideoWriter> video_writer = std::make_shared<cv::VideoWriter>();
    RET_CHECK(!video_writer->isOpened());

    {
        std::weak_ptr<cv::VideoWriter> video_writer_weak = video_writer;
        std::weak_ptr<std::mutex> video_writer_mutex_weak = video_writer_mutex;
        auto cb = [video_writer_weak = std::move(video_writer_weak), video_writer_mutex_weak = std::move(video_writer_mutex_weak)](const mediapipe::Packet &packet) -> absl::Status {
            std::shared_ptr<cv::VideoWriter> video_writer = video_writer_weak.lock();
            std::shared_ptr<std::mutex> video_writer_mutex = video_writer_mutex_weak.lock();
            if (!video_writer || !video_writer_mutex) {
                LOG(ERROR) << "Video writer released before callback.";
                return absl::OkStatus();
            }

            const mediapipe::ImageFrame& output_frame = packet.Get<mediapipe::ImageFrame>();

            // Convert back to opencv for display or saving.
            cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
            assert(output_frame_mat.data != nullptr);
            cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

            {
                assert(video_writer->isOpened());
                std::lock_guard<std::mutex> lock(*video_writer_mutex);

                video_writer->write(output_frame_mat);
            }

            return absl::OkStatus();
        };
        MP_RETURN_IF_ERROR(graph.ObserveOutputStream("out", cb));
    }

    LOG(INFO) << "Making video stream from: " << video_path;
    cv::VideoCapture capture;
    capture.open(std::move(video_path));
    RET_CHECK(capture.isOpened());

    LOG(INFO) << "Start running the MediaPipe graph.";
    MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller, graph.AddOutputStreamPoller("out"));
    MP_RETURN_IF_ERROR(graph.StartRun({}));

    bool video_writer_opened = false;

    LOG(INFO) << "Start grabbing and processing frames.";
    while (true) {
        // Capture opencv camera or video frame.
        cv::Mat input_frame_bgr_mat;
        capture >> input_frame_bgr_mat;
        if (input_frame_bgr_mat.empty()) {
            LOG(INFO) << "Empty frame, end of video reached.";
            break;
        }

        // double check lock pattern
        if (!video_writer_opened) {
            std::lock_guard<std::mutex> lock(*video_writer_mutex);
            if (!video_writer_opened) {
                video_writer->open(output_path, mediapipe::fourcc('a', 'v', 'c', '1'), 30, cv::Size(input_frame_bgr_mat.cols, input_frame_bgr_mat.rows));
                assert(video_writer->isOpened());
                video_writer_opened = true;
            }
        }

        // Wrap Mat into an ImageFrame.
        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGB, input_frame_bgr_mat.cols, input_frame_bgr_mat.rows, mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        // Convert to RGB, and copy into input frame.
        cv::cvtColor(input_frame_bgr_mat, input_frame_mat, cv::COLOR_BGR2RGB);

        // Send image packet into the graph.
        size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream("in", mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(frame_timestamp_us))));

        if (gSignalStatus) {
            LOG(INFO) << "Interrupt signal received.";
            break;
        }
    }

    MP_RETURN_IF_ERROR(graph.CloseInputStream("in"));
    MP_RETURN_IF_ERROR(graph.WaitUntilDone());

    {
        std::lock_guard<std::mutex> lock(*video_writer_mutex);
        if (video_writer->isOpened()) {
            video_writer->release();
        }
    }

    return absl::OkStatus();
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


