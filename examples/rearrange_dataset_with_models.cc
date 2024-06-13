#include <filesystem>
#include <inja/inja.hpp>

#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "glog/logging.h"

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"
#include "mediapipe/tasks/cc/components/containers/embedding_result.h"

#include "utils/image_database.h"

std::filesystem::path kImageDatabaseEmbeddingModelPath = "mobilenet_v3_large.tflite";
int kImageDatabaseEmbeddingSize = 1280;
std::filesystem::path kImageDatabasePath = "unportable_rearrenge_image_database.db";

ABSL_FLAG(std::vector<std::string>, models, std::vector<std::string>(), "Comma separated list of models to use for pre-classification.");
ABSL_FLAG(std::string, src_dir, "", "Directory containing images to be rearranged.");
ABSL_FLAG(std::string, dst_dir, "", "Directory to store rearranged images.");

// store_true
ABSL_FLAG(bool, resume, false, "Resume the process.");

const char kOutputDirTag[] = "OUTPUT_DIR";
const char kEmbeddingsTag[] = "EMBEDDINGS";
const char kClassificationsTag[] = "CLASSIFICATIONS";
const char kFilePathTag[] = "FILE_PATH";

class RearrangeDatasetCalculator : public mediapipe::CalculatorBase {
    public:
        static mediapipe::Status GetContract(mediapipe::CalculatorContract* cc) {
            cc->InputSidePackets().Tag(kOutputDirTag).Set<std::filesystem::path>();

            cc->Inputs().Tag(kEmbeddingsTag).Set<mediapipe::tasks::components::containers::proto::EmbeddingResult>();
            RET_CHECK_EQ(cc->Inputs().NumEntries(kEmbeddingsTag), 1);

            const int num_classifications_streams = cc->Inputs().NumEntries(kClassificationsTag);
            for (int i = 0; i < num_classifications_streams; ++i) {
                cc->Inputs().Get(kClassificationsTag, i).Set<mediapipe::tasks::components::containers::proto::ClassificationResult>();
            }

            cc->Inputs().Tag(kFilePathTag).Set<std::filesystem::path>();
            RET_CHECK_EQ(cc->Inputs().NumEntries(kFilePathTag), 1);

            return mediapipe::OkStatus();
        }

        mediapipe::Status Process(mediapipe::CalculatorContext* cc) final {
            const std::filesystem::path dst_dir = cc->InputSidePackets().Tag(kOutputDirTag).Get<std::filesystem::path>();
            if (dst_dir.empty()) {
                return mediapipe::InternalError("Output directory is not set.");
            }

            // get embedding result
            const mediapipe::Packet& embeddings_packet = cc->Inputs().Tag(kEmbeddingsTag).Value();
            if (embeddings_packet.IsEmpty()) {
                return mediapipe::tool::StatusStop();
            }
            const mediapipe::tasks::components::containers::proto::EmbeddingResult& embedding_result = embeddings_packet.Get<mediapipe::tasks::components::containers::proto::EmbeddingResult>();

            // get file path
            const mediapipe::Packet& file_path_packet = cc->Inputs().Tag(kFilePathTag).Value();
            if (file_path_packet.IsEmpty()) {
                return mediapipe::tool::StatusStop();
            }
            const std::filesystem::path& file_path = file_path_packet.Get<std::filesystem::path>();

            // get classification results   
            const mediapipe::Packet& classification_packet = cc->Inputs().Tag(kClassificationsTag).Value();
            std::vector<mediapipe::tasks::components::containers::proto::ClassificationResult> classification_results;
            const int num_classifications_streams = cc->Inputs().NumEntries(kClassificationsTag);
            for (int i = 0; i < num_classifications_streams; ++i) {
                const mediapipe::Packet& classification_packet = cc->Inputs().Get(kClassificationsTag, i).Value();
                if (!classification_packet.IsEmpty()) {
                    classification_results.push_back(classification_packet.Get<mediapipe::tasks::components::containers::proto::ClassificationResult>());
                }
            }

            // get max similarity from embedding, and insert the embedding to database
            const google::protobuf::RepeatedField<float>& pb_embedding_floats = embedding_result.embeddings(0).float_embedding().values();
            std::vector<float> embedding(pb_embedding_floats.begin(), pb_embedding_floats.end());
            ImageDatabase& image_database = ImageDatabase::GetInstance();
            MP_ASSIGN_OR_RETURN(float similarity, image_database.SearchMaxSimilarity(embedding));
            MP_RETURN_IF_ERROR(image_database.Insert(file_path, embedding));

            std::string dst_dirname;

            // over 0.85 is almost duplicated image
            if (similarity > 0.85) {
                dst_dirname = "duplicated";
            } else {
                for (const mediapipe::tasks::components::containers::proto::ClassificationResult& classification_result : classification_results) {
                    MP_ASSIGN_OR_RETURN(std::string dirname_component, GetDirnameComponent(classification_result));
                    if (!dst_dirname.empty()) {
                        dst_dirname += "__";
                    }
                    dst_dirname += dirname_component;
                }
            }

            std::filesystem::path dst_path = dst_dir / dst_dirname / file_path.filename();
            std::filesystem::create_directories(dst_path.parent_path());
            std::filesystem::rename(file_path, dst_path);

            return mediapipe::OkStatus();
        }

    private:
        static mediapipe::StatusOr<std::string> GetDirnameComponent(const mediapipe::tasks::components::containers::proto::ClassificationResult& classification_result) {
            std::string dirname;
            for (const auto& classifications : classification_result.classifications()) {
                const auto num_classifications = classifications.classification_list().classification_size();
                for (const auto& classification : classifications.classification_list().classification()) {
                    if (!dirname.empty()) {
                        dirname += "_";
                    }
                    dirname += classification.label();
                    float score = classification.score();
                    dirname += "_" + GetPaddedScore(score);
                }
            }
            return dirname;
        }

        static std::string GetPaddedScore(float score) {
            // -0.92 -> 00
            // 0.92 -> 09
            // 1.92 -> 19
            // 10.92 -> 99
            if (score < 0.0) {
                score = 0.0;
            } else if (score > 10.0) {
                score = 9.9;
            }
            int score_int = static_cast<int>(score * 10);
            std::string padding = "00";
            std::string score_str = std::to_string(score_int);
            return padding.substr(score_str.size()) + score_str;
        }
};

REGISTER_CALCULATOR(RearrangeDatasetCalculator);

absl::Status RunMediapipe(const std::vector<std::filesystem::path>& model_paths, const std::filesystem::path& src_dir, const std::filesystem::path& dst_dir) {

    // parse calculator graph
    inja::Environment env;
    inja::Template calculator_graph_template = env.parse(R"pb(
        input_side_packet: "image_dir"
        input_side_packet: "output_dir"

        node {
            calculator: "DecodeImageDirCalculator"
            input_side_packet: "IMAGE_DIR:image_dir"
            output_stream: "RELATIVE_FILE_PATH:relative_file_path"
            output_stream: "ABSOLUTE_FILE_PATH:file_path"
            output_stream: "FILENAME:filename"
            output_stream: "IMAGE:image_frame"
        }

        node {
            calculator: "ToImageCalculator"
            input_stream: "IMAGE:image_frame"
            output_stream: "IMAGE:image"
        }

        # Embed image
        node {
            calculator: "mediapipe.tasks.vision.image_embedder.ImageEmbedderGraph"
            input_stream: "IMAGE:image"
            output_stream: "EMBEDDINGS:embeddings"
            node_options {
                [type.googleapis.com/mediapipe.tasks.vision.image_embedder.proto.ImageEmbedderGraphOptions] {
                    base_options {
                        model_asset {
                            file_name: "{{ image_database_embedding_model_path }}"
                        }
                    }
                    embedder_options {
                        l2_normalize: true
                    }
                }
            }
        }

        {% for model_path in model_paths %}

            # Classify image
            node {
                calculator: "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph"
                input_stream: "IMAGE:image"
                output_stream: "CLASSIFICATIONS:classifications_{{ loop.index }}"
                node_options: {
                    [type.googleapis.com/mediapipe.tasks.vision.image_classifier.proto.ImageClassifierGraphOptions] {
                        base_options: {
                            model_asset: {
                                file_name: "{{ model_path }}"
                            }
                        }
                        classifier_options: {
                            max_results: 1
                        }
                    }
                }
            }

        {% endfor %}

        node {
            calculator: "RearrangeDatasetCalculator"
            input_side_packet: "OUTPUT_DIR:output_dir"
            input_stream: "EMBEDDINGS:embeddings"
            {% for model_path in model_paths %}
                input_stream: "CLASSIFICATIONS:{{loop.index}}:classifications_{{ loop.index }}"
            {% endfor %}
            input_stream: "FILE_PATH:file_path"
        }
    )pb");
    std::vector<std::string> model_path_strs;
    for (const std::filesystem::path& model_path : model_paths) {
        model_path_strs.push_back(model_path.string());
    }
    std::string calculator_graph_str = env.render(calculator_graph_template, {
            { "model_paths", model_path_strs },
            { "image_database_embedding_model_path", kImageDatabaseEmbeddingModelPath.string() },
            });
    LOG(INFO) << "Calculator graph: " << calculator_graph_str;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_str);

    // initialize calculator graph
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    MP_RETURN_IF_ERROR(graph.StartRun({
                { "image_dir", mediapipe::MakePacket<std::filesystem::path>(src_dir) },
                { "output_dir", mediapipe::MakePacket<std::filesystem::path>(dst_dir) },
                }));

    MP_RETURN_IF_ERROR(graph.WaitUntilDone());

    return absl::OkStatus();
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);

    std::vector<std::string> model_strs = absl::GetFlag(FLAGS_models);
    if (model_strs.empty()) {
        LOG(ERROR) << "No models specified.";
        return EXIT_FAILURE;
    }
    std::vector<std::filesystem::path> model_paths;
    for (const std::string& model_str : model_strs) {
        std::filesystem::path model_path = model_str;
        if (!std::filesystem::exists(model_path)) {
            LOG(ERROR) << "Model file does not exist (--models): " << model_str;
            return EXIT_FAILURE;
        }
        model_paths.push_back(model_str);
    }

    std::string src_dir_str = absl::GetFlag(FLAGS_src_dir);
    if (src_dir_str.empty()) {
        LOG(ERROR) << "No source directory specified (--src_dir).";
        return EXIT_FAILURE;
    }
    std::filesystem::path src_dir = src_dir_str;
    if (!std::filesystem::exists(src_dir)) {
        LOG(ERROR) << "Source directory does not exist (--src_dir): " << src_dir_str;
        return EXIT_FAILURE;
    }
    if (!std::filesystem::is_directory(src_dir)) {
        LOG(ERROR) << "Source path is not a directory (--src_dir): " << src_dir_str;
        return EXIT_FAILURE;
    }

    std::string dst_dir_str = absl::GetFlag(FLAGS_dst_dir);
    if (dst_dir_str.empty()) {
        LOG(ERROR) << "No destination directory specified (--dst_dir)";
        return EXIT_FAILURE;
    }
    std::filesystem::path dst_dir = dst_dir_str;
    if (!std::filesystem::exists(dst_dir)) {
        std::filesystem::create_directories(dst_dir);
    }
    if (!std::filesystem::is_directory(dst_dir)) {
        LOG(ERROR) << "Destination path is not a directory (--dst_dir): " << dst_dir_str;
        return EXIT_FAILURE;
    }

    bool resume = absl::GetFlag(FLAGS_resume);

    if (!resume) {
        if (std::filesystem::exists(kImageDatabasePath)) {
            std::cout << "Database already exists. If you want to resume set flag --resume " << std::endl;
            return EXIT_FAILURE;
        }
        if (!std::filesystem::is_empty(dst_dir)) {
            std::cout << "Error: Destination directory is not empty. If you want to resume set flag --resume " << std::endl;
            return EXIT_FAILURE;
        }
    }

    ImageDatabase::Initialize(kImageDatabaseEmbeddingModelPath, kImageDatabaseEmbeddingSize, kImageDatabasePath);


    absl::Status mediapipe_status = RunMediapipe(model_paths, src_dir, dst_dir);
    if (!mediapipe_status.ok()) {
        LOG(ERROR) << "Error running mediapipe: " << mediapipe_status.message();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

