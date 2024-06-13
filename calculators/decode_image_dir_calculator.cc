#include <filesystem>
#include <unordered_set>
#include <string>
#include <memory>

#include "utils/recursive_dir_iter.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_imgcodecs_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/tool/status_util.h"

const std::unordered_set<std::string> kSupportedImageExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"};

// input side packet
const char kImageDirTag[] = "IMAGE_DIR";

// output packet
const char kRelativeFilePathTag[] = "RELATIVE_FILE_PATH";
const char kAbsoluteFilePathTag[] = "ABSOLUTE_FILE_PATH";
const char kFilenameTag[] = "FILENAME";
const char kImageTag[] = "IMAGE";

class DecodeImageDirCalculator : public mediapipe::CalculatorBase {
    private:
        bool open_ = false;
        std::filesystem::path image_dir_;
        RecursiveDirIterable image_dir_iterable_;
        RecursiveDirIterator image_dir_iter_;
        mediapipe::Timestamp prev_timestamp_ = mediapipe::Timestamp::Unset();

    public:
        static mediapipe::Status GetContract(mediapipe::CalculatorContract* cc) {
            cc->InputSidePackets().Tag(kImageDirTag).Set<std::filesystem::path>();
            cc->Outputs().Tag(kRelativeFilePathTag).Set<std::filesystem::path>();
            cc->Outputs().Tag(kAbsoluteFilePathTag).Set<std::filesystem::path>();
            cc->Outputs().Tag(kFilenameTag).Set<std::string>();
            cc->Outputs().Tag(kImageTag).Set<mediapipe::ImageFrame>();
            return mediapipe::OkStatus();
        }

        mediapipe::Status Open(mediapipe::CalculatorContext* cc) final {
            assert(!open_);
            const std::filesystem::path& image_relative_dir = cc->InputSidePackets().Tag(kImageDirTag).Get<std::filesystem::path>();

            // to absolute canonical path
            const std::filesystem::path image_dir = std::filesystem::canonical(image_relative_dir);

            if (!std::filesystem::exists(image_dir)) {
                return mediapipe::InternalError("Input image directory does not exist.");
            }
            if (!std::filesystem::is_directory(image_dir)) {
                return mediapipe::InternalError("Input image directory is not a directory.");
            }

            image_dir_ = image_dir;
            image_dir_iterable_ = RecursiveDirIterable(image_dir);
            image_dir_iter_ = image_dir_iterable_.begin();
            prev_timestamp_ = mediapipe::Timestamp(0);

            open_ = true;
            return mediapipe::OkStatus();
        }

        mediapipe::Status Process(mediapipe::CalculatorContext* cc) final {
            assert(open_);
            if (image_dir_iter_ == image_dir_iterable_.end()) {
                return mediapipe::tool::StatusStop();
            }

            mediapipe::Timestamp timestamp = prev_timestamp_ + 1;
            prev_timestamp_ = timestamp;

            const std::filesystem::path& image_org_path = *image_dir_iter_;
            ++image_dir_iter_;
            if (!std::filesystem::is_regular_file(image_org_path)) {
                // just ignore
                LOG(WARNING) << "Not a regular file, ignore it: " << image_org_path.string();
                return mediapipe::OkStatus();
            }
            std::string extension = image_org_path.extension().string();
            if (!kSupportedImageExtensions.count(extension)) {
                // just ignore
                LOG(WARNING) << "Unsupported image extension, ignore it: " << image_org_path.string();
                return mediapipe::OkStatus();
            }

            // use relative path for packet
            const std::filesystem::path image_path = image_org_path.lexically_proximate(image_dir_);

            // read image file with opencv
            cv::Mat bgr_image = cv::imread(image_org_path.string(), cv::IMREAD_COLOR);
            if (bgr_image.empty()) {
                return mediapipe::InternalError("Failed to read image: " + image_org_path.string());
            }

            int num_channels = bgr_image.channels();
            mediapipe::ImageFormat::Format format;
            if (num_channels == 1) {
                format = mediapipe::ImageFormat::GRAY8;
            } else if (num_channels ==3) {
                format = mediapipe::ImageFormat::SRGB;
            } else if (num_channels == 4) {
                format = mediapipe::ImageFormat::SRGBA;
            } else {
                return mediapipe::InternalError("Unsupported image num_channels: " + image_org_path.string());
            }

            int frame_width = bgr_image.cols;
            int frame_height = bgr_image.rows;
            std::unique_ptr<mediapipe::ImageFrame> image_frame = std::make_unique<mediapipe::ImageFrame>(format, frame_width, frame_height, /*alignment_boundary=*/1);
            cv::cvtColor(bgr_image, mediapipe::formats::MatView(image_frame.get()), cv::COLOR_BGR2RGB);

            // absolute path for output packet
            const std::filesystem::path absolute_image_path = std::filesystem::canonical(image_org_path);

            cc->Outputs().Tag(kRelativeFilePathTag).AddPacket(mediapipe::MakePacket<std::filesystem::path>(image_path).At(timestamp));
            cc->Outputs().Tag(kAbsoluteFilePathTag).AddPacket(mediapipe::MakePacket<std::filesystem::path>(absolute_image_path).At(timestamp));
            cc->Outputs().Tag(kFilenameTag).AddPacket(mediapipe::MakePacket<std::string>(image_org_path.filename().string()).At(timestamp));
            cc->Outputs().Tag(kImageTag).Add(image_frame.release(), timestamp);

            return mediapipe::OkStatus();
        }

        absl::Status Close(mediapipe::CalculatorContext* cc) override {
            assert(open_);
            image_dir_.clear();
            image_dir_iterable_ = RecursiveDirIterable();
            image_dir_iter_ = RecursiveDirIterator();
            prev_timestamp_ = mediapipe::Timestamp::Unset();
            open_ = false;

            return absl::OkStatus();
        }
};

REGISTER_CALCULATOR(DecodeImageDirCalculator);

