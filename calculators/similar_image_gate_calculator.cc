#include <string>
#include <optional>
#include <utility>

#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"
#include "calculators/similar_image_gate_calculator.pb.h"
#include "utils/image_database.h"

constexpr char kEmbeddingsTag[] = "EMBEDDINGS";

class SimilarImageGateCalculator : public mediapipe::CalculatorBase {
    private:
        float similarity_threshold_ = 0.8;

        mediapipe::Status LoadOptions(mediapipe::CalculatorContext& cc) {
            const SimilarImageGateCalculatorOptions& options = cc.Options<SimilarImageGateCalculatorOptions>();
            similarity_threshold_ = options.similarity_threshold();
            return mediapipe::OkStatus();
        }

    public:
        static mediapipe::Status GetContract(mediapipe::CalculatorContract* cc) {
            // input embeddings
            RET_CHECK(cc->Inputs().HasTag(kEmbeddingsTag));
            RET_CHECK_EQ(cc->Inputs().NumEntries(kEmbeddingsTag), 1);

            // output embeddings
            RET_CHECK(cc->Outputs().HasTag(kEmbeddingsTag));
            RET_CHECK_EQ(cc->Outputs().NumEntries(kEmbeddingsTag), 1);

            // unnamed input image
            const int num_data_streams = cc->Inputs().NumEntries("");
            for (int i = 0; i < num_data_streams; ++i) {
                cc->Inputs().Get("", i).SetAny();
                cc->Outputs().Get("", i).SetSameAs(&cc->Inputs().Get("", i));
            }

            return mediapipe::OkStatus();
        }

        mediapipe::Status Open(mediapipe::CalculatorContext* cc) final {
            MP_RETURN_IF_ERROR(LoadOptions(*cc));

            return mediapipe::OkStatus();
        }

        mediapipe::Status Process(mediapipe::CalculatorContext* cc) final {
            MP_RETURN_IF_ERROR(LoadOptions(*cc));

            const mediapipe::Packet& embeddings_packet = cc->Inputs().Tag(kEmbeddingsTag).Value();
            if (embeddings_packet.IsEmpty()) {
                // if embeddings or image is empty, nothing to do
                return mediapipe::OkStatus();
            }

            auto& embedding_result = embeddings_packet.Get<mediapipe::tasks::components::containers::proto::EmbeddingResult>();
            assert(embedding_result.embeddings_size() == 1);

            const google::protobuf::RepeatedField<float>& pb_embedding_floats = embedding_result.embeddings(0).float_embedding().values();
            std::vector<float> embedding(pb_embedding_floats.begin(), pb_embedding_floats.end());

            ImageDatabase& image_database = ImageDatabase::GetInstance();

            absl::StatusOr<float> similarity_or_status = image_database.SearchMaxSimilarity(embedding);
            if (!similarity_or_status.ok()) {
                LOG(ERROR) << "Error: " << similarity_or_status.status().message();
                return mediapipe::InternalError("Image database search error: " + std::string(similarity_or_status.status().message()));
            }
            float similarity = similarity_or_status.value();
            if (similarity > similarity_threshold_) {
                LOG(INFO) << "Almost Similar image found, skipping";
                return mediapipe::OkStatus();
            }

            // forward embeddings
            cc->Outputs().Tag(kEmbeddingsTag).AddPacket(embeddings_packet);

            // forward other unnamed input image
            const int num_data_streams = cc->Inputs().NumEntries("");
            for (int i = 0; i < num_data_streams; ++i) {
                cc->Outputs().Get("", i).AddPacket(cc->Inputs().Get("", i).Value());
            }

            return mediapipe::OkStatus();
        }
};

REGISTER_CALCULATOR(SimilarImageGateCalculator);

