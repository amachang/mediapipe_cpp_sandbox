#include "calculators/detection_to_largest_square_rect_calculator.pb.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";

class DetectionToLargestSquareRectCalculator: public mediapipe::CalculatorBase {
    private:
        std::optional<double> relative_margin_;

        absl::Status LoadOptions(const mediapipe::CalculatorContext& cc) {
            const DetectionToLargestSquareRectCalculatorOptions& options = cc.Options<DetectionToLargestSquareRectCalculatorOptions>();
            relative_margin_ = options.relative_margin();

            return absl::OkStatus();
        }

    public:
        static mediapipe::Status GetContract(mediapipe::CalculatorContract* cc) {
            RET_CHECK(cc->Inputs().HasTag(kDetectionTag));
            RET_CHECK(cc->Inputs().HasTag(kImageSizeTag));

            RET_CHECK_EQ(cc->Inputs().NumEntries(kDetectionTag), 1);
            RET_CHECK_EQ(cc->Inputs().NumEntries(kImageSizeTag), 1);
            RET_CHECK_EQ(cc->Outputs().NumEntries(), 1);

            cc->Inputs().Tag(kDetectionTag).Set<mediapipe::Detection>();
            cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();
            cc->Outputs().Index(0).Set<mediapipe::NormalizedRect>();

            return mediapipe::OkStatus();
        }

        mediapipe::Status Open(mediapipe::CalculatorContext* cc) final {
            MP_RETURN_IF_ERROR(LoadOptions(*cc));

            return mediapipe::OkStatus();
        }

        mediapipe::Status Process(mediapipe::CalculatorContext* cc) final {
            assert(relative_margin_.has_value());

            const mediapipe::Packet& detection_packet = cc->Inputs().Tag(kDetectionTag).Value();
            if (detection_packet.IsEmpty()) {
                return mediapipe::OkStatus();
            }
            const mediapipe::Packet& image_size_packet = cc->Inputs().Tag(kImageSizeTag).Value();
            RET_CHECK(!image_size_packet.IsEmpty()) << "Image size packet must be given";

            const mediapipe::Detection& detection = detection_packet.Get<mediapipe::Detection>();
            const mediapipe::LocationData& location_data = detection.location_data();

            const std::pair<int, int>& image_size = image_size_packet.Get<std::pair<int, int>>();
            double image_width = static_cast<double>(image_size.first);
            double image_height = static_cast<double>(image_size.second);

            double xmin = 0.0;
            double ymin = 0.0;
            double xmax = 0.0;
            double ymax = 0.0;

            double box_xmin = 0.0, box_ymin = 0.0, box_xmax = 0.0, box_ymax = 0.0;
            if (location_data.format() == mediapipe::LocationData::RELATIVE_BOUNDING_BOX) {
                const mediapipe::LocationData::RelativeBoundingBox& relative_bounding_box = location_data.relative_bounding_box();
                box_xmin = relative_bounding_box.xmin() * image_width;
                box_ymin = relative_bounding_box.ymin() * image_height;
                box_xmax = (relative_bounding_box.xmin() + relative_bounding_box.width()) * image_width;
                box_ymax = (relative_bounding_box.ymin() + relative_bounding_box.height()) * image_height;
            } else if (location_data.format() == mediapipe::LocationData::BOUNDING_BOX) {
                const mediapipe::LocationData::BoundingBox& bounding_box = location_data.bounding_box();
                box_xmin = static_cast<double>(bounding_box.xmin());
                box_ymin = static_cast<double>(bounding_box.ymin());
                box_xmax = static_cast<double>(bounding_box.xmin() + bounding_box.width());
                box_ymax = static_cast<double>(bounding_box.ymin() + bounding_box.height());
            } else {
                return mediapipe::UnimplementedError("LocationData format not supported");
            }

            if (box_xmin < xmin) {
                xmin = box_xmin;
            }
            if (box_ymin < ymin) {
                ymin = box_ymin;
            }
            if (box_xmax > xmax) {
                xmax = box_xmax;
            }
            if (box_ymax > ymax) {
                ymax = box_ymax;
            }

            for (const mediapipe::LocationData::RelativeKeypoint& keypoint: location_data.relative_keypoints()) {
                double keypoint_x = keypoint.x() * image_width;
                double keypoint_y = keypoint.y() * image_height;
                if (keypoint_x < xmin) {
                    xmin = keypoint_x;
                }
                if (keypoint_y < ymin) {
                    ymin = keypoint_y;
                }
                if (keypoint_x > xmax) {
                    xmax = keypoint_x;
                }
                if (keypoint_y > ymax) {
                    ymax = keypoint_y;
                }
            }

            xmin = std::max(0.0, std::min(image_width, xmin));
            ymin = std::max(0.0, std::min(image_height, ymin));
            xmax = std::max(0.0, std::min(image_width, xmax));
            ymax = std::max(0.0, std::min(image_height, ymax));

            double width = xmax - xmin;
            double height = ymax - ymin;

            bool is_width_longer = width > height;
            double shorter_side = is_width_longer ? height : width;
            double longer_side = is_width_longer ? width : height;
            double difference = longer_side - shorter_side;
            double image_shorter_side_length = is_width_longer ? image_height : image_width;
            double shorter_side_min = is_width_longer ? ymin : xmin;
            double shorter_side_max = is_width_longer ? ymax : xmax;
            double longer_side_min = is_width_longer ? xmin : ymin;
            double longer_side_max = is_width_longer ? xmax : ymax;

            double distance_to_shorter_start_edge = shorter_side_min;
            double distance_to_shorter_end_edge = image_shorter_side_length - shorter_side_max;
            double affordable_extra_shorter_side = distance_to_shorter_start_edge + distance_to_shorter_end_edge;

            double being_subtracted_longer_side = 0.0;
            double being_added_shorter_side = difference;
            if (affordable_extra_shorter_side < difference) {
                being_subtracted_longer_side = difference - affordable_extra_shorter_side;
                being_added_shorter_side = affordable_extra_shorter_side;
            }

            longer_side_min += being_subtracted_longer_side / 2.0;
            longer_side_max -= being_subtracted_longer_side / 2.0;
            shorter_side_min -= being_added_shorter_side / 2.0;
            shorter_side_max += being_added_shorter_side / 2.0;
            if (shorter_side_min < 0.0) {
                shorter_side_max -= shorter_side_min;
                shorter_side_min = 0.0;
            }
            if (shorter_side_max > image_shorter_side_length) {
                shorter_side_min -= shorter_side_max - image_shorter_side_length;
                shorter_side_max = image_shorter_side_length;
            }

            if (is_width_longer) {
                xmin = longer_side_min;
                ymin = shorter_side_min;
                xmax = longer_side_max;
                ymax = shorter_side_max;
            } else {
                xmin = shorter_side_min;
                ymin = longer_side_min;
                xmax = shorter_side_max;
                ymax = longer_side_max;
            }

            // both are equal, but we use the shorter side
            // just for floating point calculation errors
            double before_margin_square_side = std::min(xmax - xmin, ymax - ymin);

            // apply margin
            double margin = before_margin_square_side * relative_margin_.value();
            margin = std::min(margin, xmin);
            margin = std::min(margin, ymin);
            margin = std::min(margin, image_width - xmax);
            margin = std::min(margin, image_height - ymax);
            ymin -= margin;
            ymax += margin;
            xmin -= margin;
            xmax += margin;
            double square_side = std::min(xmax - xmin, ymax - ymin);

            std::unique_ptr<mediapipe::NormalizedRect> normalized_rect = std::make_unique<mediapipe::NormalizedRect>();
            normalized_rect->set_x_center((xmin + xmax) / 2.0 / image_width);
            normalized_rect->set_y_center((ymin + ymax) / 2.0 / image_height);
            normalized_rect->set_width(square_side / image_width);
            normalized_rect->set_height(square_side / image_height);

            cc->Outputs().Index(0).AddPacket(mediapipe::Adopt(normalized_rect.release()).At(cc->InputTimestamp()));

            return mediapipe::OkStatus();
        }
};

REGISTER_CALCULATOR(DetectionToLargestSquareRectCalculator);

