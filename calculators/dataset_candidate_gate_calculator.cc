#include <iostream>
#include <string>
#include <optional>
#include <utility>
#include <tuple>
#include <vector>
#include <memory>
#include <filesystem>

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"

constexpr char kEmbeddingsTag[] = "EMBEDDINGS";
constexpr char kClassificationsTag[] = "CLASSIFICATIONS";
constexpr char kImageTag[] = "IMAGE";
constexpr char kLabelScoreEmbeddingImageTag[] = "LABEL_SCORE_EMBEDDING_IMAGE";

class DatasetCandidateGateCalculator : public mediapipe::CalculatorBase {
    private:
        mediapipe::Status LoadOptions(mediapipe::CalculatorContext& cc) {
            return mediapipe::OkStatus();
        }

    public:

        static mediapipe::Status GetContract(mediapipe::CalculatorContract* cc) {
            RET_CHECK(cc->Inputs().HasTag(kClassificationsTag));
            RET_CHECK(cc->Inputs().HasTag(kImageTag));
            RET_CHECK(cc->Outputs().HasTag(kLabelScoreEmbeddingImageTag));

            RET_CHECK_EQ(cc->Inputs().NumEntries(kEmbeddingsTag), 1);
            RET_CHECK_EQ(cc->Inputs().NumEntries(kClassificationsTag), 1);
            RET_CHECK_EQ(cc->Inputs().NumEntries(kImageTag), 1);
            RET_CHECK_EQ(cc->Outputs().NumEntries(kLabelScoreEmbeddingImageTag), 1);

            cc->Inputs().Tag(kEmbeddingsTag).Set<mediapipe::tasks::components::containers::proto::EmbeddingResult>();
            cc->Inputs().Tag(kClassificationsTag).Set<mediapipe::tasks::components::containers::proto::ClassificationResult>();
            cc->Inputs().Tag(kImageTag).Set<mediapipe::Image>();
            cc->Outputs().Tag(kLabelScoreEmbeddingImageTag).Set<std::tuple<std::vector<std::pair<std::string, double>>, std::vector<float>, std::shared_ptr<const mediapipe::ImageFrame>>>();

            return mediapipe::OkStatus();
        }

        mediapipe::Status Open(mediapipe::CalculatorContext* cc) final {
            MP_RETURN_IF_ERROR(LoadOptions(*cc));

            return mediapipe::OkStatus();
        }

        mediapipe::Status Process(mediapipe::CalculatorContext* cc) final {
            MP_RETURN_IF_ERROR(LoadOptions(*cc));

            const mediapipe::Packet& embeddings_packet = cc->Inputs().Tag(kEmbeddingsTag).Value();
            const mediapipe::Packet& classifications_packet = cc->Inputs().Tag(kClassificationsTag).Value();
            const mediapipe::Packet& image_packet = cc->Inputs().Tag(kImageTag).Value();
            if (embeddings_packet.IsEmpty() && classifications_packet.IsEmpty() && image_packet.IsEmpty()) {
                return mediapipe::OkStatus();
            }

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

            const mediapipe::tasks::components::containers::proto::EmbeddingResult& embedding_result = embeddings_packet.Get<mediapipe::tasks::components::containers::proto::EmbeddingResult>();
            assert(embedding_result.embeddings_size() == 1);
            const mediapipe::tasks::components::containers::proto::Embedding& pb_embedding = embedding_result.embeddings(0);
            const google::protobuf::RepeatedField<float>& pb_embedding_floats = pb_embedding.float_embedding().values();
            std::vector<float> embedding(pb_embedding_floats.begin(), pb_embedding_floats.end());

            const mediapipe::Image& image = image_packet.Get<mediapipe::Image>();
            std::shared_ptr<const mediapipe::ImageFrame> image_frame = image.GetImageFrameSharedPtr();
            std::tuple<std::vector<std::pair<std::string, double>>, std::vector<float>, std::shared_ptr<const mediapipe::ImageFrame>> output = 
                std::make_tuple(label_and_score_list, embedding, image_frame);
            mediapipe::Packet output_packet = mediapipe::MakePacket<std::tuple<std::vector<std::pair<std::string, double>>, std::vector<float>, std::shared_ptr<const mediapipe::ImageFrame>>>(output).At(classifications_packet.Timestamp());

            cc->Outputs().Tag(kLabelScoreEmbeddingImageTag).AddPacket(output_packet);

            return mediapipe::OkStatus();
        }
};

REGISTER_CALCULATOR(DatasetCandidateGateCalculator);
