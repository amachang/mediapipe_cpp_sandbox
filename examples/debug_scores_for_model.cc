#include <filesystem>
#include <iostream>
#include <string>
#include <memory>

#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "glog/logging.h"

#include <inja/inja.hpp>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"

ABSL_FLAG(std::string, model_path, "", "model path");
ABSL_FLAG(std::string, image_path, "", "image path");

const char kImagePathTag[] = "IMAGE_PATH";
const char kRgbImageFrameTag[] = "RGB_IMAGE_FRAME";
const char kClassificationsTag[] = "CLASSIFICATIONS";

class ReadSingleImageCalculator : public mediapipe::CalculatorBase {
    public:
        static mediapipe::Status GetContract(mediapipe::CalculatorContract* cc) {
            cc->InputSidePackets().Tag(kImagePathTag).Set<std::filesystem::path>();
            cc->Outputs().Tag(kRgbImageFrameTag).Set<mediapipe::ImageFrame>();
            return mediapipe::OkStatus();
        }

        mediapipe::Status Open(mediapipe::CalculatorContext* cc) override {
            image_path_ = cc->InputSidePackets().Tag(kImagePathTag).Get<std::filesystem::path>();
            processed_ = false;
            return mediapipe::OkStatus();
        }

        mediapipe::Status Process(mediapipe::CalculatorContext* cc) override {
            if (processed_) {
                return mediapipe::tool::StatusStop();
            }
            processed_ = true;

            mediapipe::Timestamp timestamp = mediapipe::Timestamp(1);

            cv::Mat bgr_image = cv::imread(image_path_.string());
            if (bgr_image.empty()) {
                return mediapipe::InternalError("Failed to read image.: " + image_path_.string());
            }
            assert(bgr_image.channels() == 3);
            const mediapipe::ImageFormat::Format format = mediapipe::ImageFormat::SRGB;
            int frame_width = bgr_image.cols;
            int frame_height = bgr_image.rows;
            std::unique_ptr<mediapipe::ImageFrame> image_frame = std::make_unique<mediapipe::ImageFrame>(format, frame_width, frame_height, /*alignment_boundary=*/1);
            cv::cvtColor(bgr_image, mediapipe::formats::MatView(image_frame.get()), cv::COLOR_BGR2RGB);

            cc->Outputs().Tag(kRgbImageFrameTag).Add(image_frame.release(), timestamp);
            return mediapipe::OkStatus();
        }

        mediapipe::Status Close(mediapipe::CalculatorContext* cc) override {
            processed_ = false;
            image_path_.clear();
            return mediapipe::OkStatus();
        }
    private:
        bool processed_;
        std::filesystem::path image_path_;
};

REGISTER_CALCULATOR(ReadSingleImageCalculator);

class ClassificationsPrinterCalculator : public mediapipe::CalculatorBase {
    public:
        static mediapipe::Status GetContract(mediapipe::CalculatorContract* cc) {
            cc->Inputs().Tag(kClassificationsTag).Set<mediapipe::tasks::components::containers::proto::ClassificationResult>();
            RET_CHECK_EQ(cc->Inputs().NumEntries(kClassificationsTag), 1);
            return mediapipe::OkStatus();
        }

        mediapipe::Status Open(mediapipe::CalculatorContext* cc) override {
            return mediapipe::OkStatus();
        }

        mediapipe::Status Process(mediapipe::CalculatorContext* cc) override {
            const auto& classification_result = cc->Inputs().Tag(kClassificationsTag).Get<mediapipe::tasks::components::containers::proto::ClassificationResult>();
            for (const auto& classifications : classification_result.classifications()) {
                const auto num_classifications = classifications.classification_list().classification_size();
                for (const auto& classification : classifications.classification_list().classification()) {
                    std::cout << classification.label() << " " << classification.score() << std::endl;
                }
            }
            return mediapipe::OkStatus();
        }
};

REGISTER_CALCULATOR(ClassificationsPrinterCalculator);

absl::Status RunCase0(const std::filesystem::path& model_path, const std::filesystem::path& image_path) {
    assert(std::filesystem::exists(model_path) && std::filesystem::is_regular_file(model_path));
    assert(std::filesystem::exists(image_path) && std::filesystem::is_regular_file(image_path));

    inja::Environment env;
    inja::Template calculator_graph_template = env.parse(R"pb(
        input_side_packet: "image_path"

        # Input 
        node {
            calculator: "ReadSingleImageCalculator"
            input_side_packet: "IMAGE_PATH:image_path"
            output_stream: "RGB_IMAGE_FRAME:original_rgb_image"
        }

        node: {
            calculator: "ImageTransformationCalculator"
            input_stream: "IMAGE:original_rgb_image"
            output_stream: "IMAGE:square_input_image"
            node_options: {
                [type.googleapis.com/mediapipe.ImageTransformationCalculatorOptions] {
                    output_width: 300
                    output_height: 300
                    scale_mode: FIT
                }
            }
        }

        node {
            calculator: "ToImageCalculator"
            input_stream: "IMAGE:square_input_image"
            output_stream: "IMAGE:image_used_by_vision_tasks"
        }

        # Classify image
        node {
            calculator: "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph"
            input_stream: "IMAGE:image_used_by_vision_tasks"
            output_stream: "CLASSIFICATIONS:classifications"
            node_options: {
                [type.googleapis.com/mediapipe.tasks.vision.image_classifier.proto.ImageClassifierGraphOptions] {
                    base_options: {
                        model_asset: {
                            file_name: "{{ model_path }}" # TODO escape path
                        }
                    }
                }
            }
        }

        # Output
        node {
            calculator: "ClassificationsPrinterCalculator"
            input_stream: "CLASSIFICATIONS:classifications"
        }
    )pb");
    std::string calculator_graph_str = env.render(calculator_graph_template, { {"model_path", model_path.string()} });
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_str);
    mediapipe::CalculatorGraph graph;
    std::cout << "--- Case0 ---" << std::endl;
    MP_RETURN_IF_ERROR(graph.Initialize(config));
    MP_RETURN_IF_ERROR(graph.StartRun({ {"image_path", mediapipe::MakePacket<std::filesystem::path>(image_path)} }));
    MP_RETURN_IF_ERROR(graph.WaitUntilDone());
    std::cout << "---" << std::endl;
    return absl::OkStatus();
}

absl::Status RunCase1(const std::filesystem::path& model_path, const std::filesystem::path& image_path) {
    assert(std::filesystem::exists(model_path) && std::filesystem::is_regular_file(model_path));
    assert(std::filesystem::exists(image_path) && std::filesystem::is_regular_file(image_path));

    inja::Environment env;
    inja::Template calculator_graph_template = env.parse(R"pb(
        input_side_packet: "image_path"

        # Input 
        node {
            calculator: "ReadSingleImageCalculator"
            input_side_packet: "IMAGE_PATH:image_path"
            output_stream: "RGB_IMAGE_FRAME:original_rgb_image"
        }

        node {
            calculator: "ToImageCalculator"
            input_stream: "IMAGE:original_rgb_image"
            output_stream: "IMAGE:image"
        }

        # Classify image
        node {
            calculator: "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph"
            input_stream: "IMAGE:image"
            output_stream: "CLASSIFICATIONS:classifications"
            node_options: {
                [type.googleapis.com/mediapipe.tasks.vision.image_classifier.proto.ImageClassifierGraphOptions] {
                    base_options: {
                        model_asset: {
                            file_name: "{{ model_path }}"
                        }
                    }
                }
            }
        }

        # Output
        node {
            calculator: "ClassificationsPrinterCalculator"
            input_stream: "CLASSIFICATIONS:classifications"
        }
    )pb");
    std::string calculator_graph_str = env.render(calculator_graph_template, { {"model_path", model_path.string()} });
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_str);
    mediapipe::CalculatorGraph graph;
    std::cout << "--- Case1 ---" << std::endl;
    MP_RETURN_IF_ERROR(graph.Initialize(config));
    MP_RETURN_IF_ERROR(graph.StartRun({ {"image_path", mediapipe::MakePacket<std::filesystem::path>(image_path)} }));
    MP_RETURN_IF_ERROR(graph.WaitUntilDone());
    std::cout << "---" << std::endl;
    return absl::OkStatus();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    absl::ParseCommandLine(argc, argv);

    const std::string model_path_str = absl::GetFlag(FLAGS_model_path);
    if (model_path_str.empty()) {
        LOG(ERROR) << "model_path is required (--model_path)";
        return EXIT_FAILURE;
    }
    const std::filesystem::path model_path = std::filesystem::path(model_path_str);

    const std::string image_path_str = absl::GetFlag(FLAGS_image_path);
    if (image_path_str.empty()) {
        LOG(ERROR) << "image_path is required (--image_path)";
        return EXIT_FAILURE;
    }
    const std::filesystem::path image_path = std::filesystem::path(image_path_str);

    absl::Status status = RunCase0(model_path, image_path);
    if (!status.ok()) {
        LOG(ERROR) << "RunCase0 failed: " << status.message();
    }
    status = RunCase1(model_path, image_path);
    if (!status.ok()) {
        LOG(ERROR) << "RunCase1 failed: " << status.message();
    }

    return EXIT_SUCCESS;
}

