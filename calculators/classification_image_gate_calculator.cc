#include <string>
#include <optional>
#include <utility>

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"

constexpr char kClassificationsTag[] = "CLASSIFICATIONS";
constexpr char kImageTag[] = "IMAGE";
constexpr char kLabelScoreImageTag[] = "LABEL_SCORE_IMAGE";

class ClassificationImageGateCalculator : public mediapipe::CalculatorBase {
    private:
        mediapipe::Status LoadOptions(mediapipe::CalculatorContext& cc) {
            // currently nothing to load
            return mediapipe::OkStatus();
        }

    public:
        static mediapipe::Status GetContract(mediapipe::CalculatorContract* cc) {
            RET_CHECK(cc->Inputs().HasTag(kClassificationsTag));
            RET_CHECK(cc->Inputs().HasTag(kImageTag));
            RET_CHECK(cc->Outputs().HasTag(kLabelScoreImageTag));

            RET_CHECK_EQ(cc->Inputs().NumEntries(kClassificationsTag), 1);
            RET_CHECK_EQ(cc->Inputs().NumEntries(kImageTag), 1);
            RET_CHECK_EQ(cc->Outputs().NumEntries(kLabelScoreImageTag), 1);

            cc->Inputs().Tag(kClassificationsTag).Set<mediapipe::tasks::components::containers::proto::ClassificationResult>();
            cc->Inputs().Tag(kImageTag).Set<mediapipe::Image>();
            cc->Outputs().Tag(kLabelScoreImageTag).Set<std::pair<std::vector<std::pair<std::string, double>>, std::shared_ptr<const mediapipe::ImageFrame>>>();

            return mediapipe::OkStatus();
        }

        mediapipe::Status Open(mediapipe::CalculatorContext* cc) final {
            MP_RETURN_IF_ERROR(LoadOptions(*cc));

            return mediapipe::OkStatus();
        }

        mediapipe::Status Process(mediapipe::CalculatorContext* cc) final {
            MP_RETURN_IF_ERROR(LoadOptions(*cc));

            const mediapipe::Packet& classifications_packet = cc->Inputs().Tag(kClassificationsTag).Value();
            const mediapipe::Packet& image_packet = cc->Inputs().Tag(kImageTag).Value();
            if (classifications_packet.IsEmpty() && image_packet.IsEmpty()) {
                return mediapipe::OkStatus();
            }
            RET_CHECK(!classifications_packet.IsEmpty() && !image_packet.IsEmpty());

            std::vector<std::pair<std::string, double>> label_and_score_list;
            const mediapipe::tasks::components::containers::proto::ClassificationResult& classification_result = classifications_packet.Get<mediapipe::tasks::components::containers::proto::ClassificationResult>();
            for (const auto& classifications : classification_result.classifications()) {
                const auto num_classifications = classifications.classification_list().classification_size();
                for (const auto& classification : classifications.classification_list().classification()) {
                    label_and_score_list.push_back(std::make_pair(classification.label(), classification.score()));
                }
            }

            if (label_and_score_list.empty()) {
                return mediapipe::OkStatus();
            }

            const mediapipe::Image& image = image_packet.Get<mediapipe::Image>();
            std::shared_ptr<const mediapipe::ImageFrame> image_frame = image.GetImageFrameSharedPtr();
            std::pair<std::vector<std::pair<std::string, double>>, std::shared_ptr<const mediapipe::ImageFrame>> output = std::make_pair(label_and_score_list, image_frame);
            mediapipe::Packet output_packet = mediapipe::MakePacket<std::pair<std::vector<std::pair<std::string, double>>, std::shared_ptr<const mediapipe::ImageFrame>>>(output).At(classifications_packet.Timestamp());

            cc->Outputs().Tag(kLabelScoreImageTag).AddPacket(output_packet);

            return mediapipe::OkStatus();
        }
};

REGISTER_CALCULATOR(ClassificationImageGateCalculator);

