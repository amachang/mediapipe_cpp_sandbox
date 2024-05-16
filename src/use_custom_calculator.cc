#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "glog/logging.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"

class CustomCalculator : public mediapipe::CalculatorBase {
    public:
        static absl::Status GetContract(mediapipe::CalculatorContract* cc) {
            std::cout << "GetContract" << std::endl;
            if (!cc->Inputs().TagMap()->SameAs(*cc->Outputs().TagMap())) {
                return absl::InvalidArgumentError("Input and output streams must use matching tags and indexes.");
            }

            for (mediapipe::CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId(); ++id) {
                cc->Inputs().Get(id).SetAny();
                cc->Outputs().Get(id).SetSameAs(&cc->Inputs().Get(id));
            }

            for (mediapipe::CollectionItemId id = cc->InputSidePackets().BeginId(); id < cc->InputSidePackets().EndId(); ++id) {
                cc->InputSidePackets().Get(id).SetAny();
            }

            if (cc->OutputSidePackets().NumEntries() != 0) {
                if (!cc->InputSidePackets().TagMap()->SameAs(*cc->OutputSidePackets().TagMap())) {
                    return absl::InvalidArgumentError("Input and output side packets must use matching tags and indexes.");
                }
                for (mediapipe::CollectionItemId id = cc->InputSidePackets().BeginId(); id < cc->InputSidePackets().EndId(); ++id) {
                    cc->OutputSidePackets().Get(id).SetSameAs(&cc->InputSidePackets().Get(id));
                }
            }

            return absl::OkStatus();
        }

        absl::Status Open(mediapipe::CalculatorContext* cc) final {
            std::cout << "Open" << std::endl;
            for (mediapipe::CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId(); ++id) {
                if (!cc->Inputs().Get(id).Header().IsEmpty()) {
                    cc->Outputs().Get(id).SetHeader(cc->Inputs().Get(id).Header());
                }
            }
            if (cc->OutputSidePackets().NumEntries() != 0) {
                for (mediapipe::CollectionItemId id = cc->InputSidePackets().BeginId(); id < cc->InputSidePackets().EndId(); ++id) {
                    cc->OutputSidePackets().Get(id).Set(cc->InputSidePackets().Get(id));
                }
            }
            cc->SetOffset(mediapipe::TimestampDiff(0));
            return absl::OkStatus();
        }

        absl::Status Process(mediapipe::CalculatorContext* cc) final {
            mediapipe::Counter* counter = cc->GetCounter("ProcessCount");
            std::cout << "Process Started: #" << counter->Get() << std::endl;
            counter->Increment();

            if (cc->Inputs().NumEntries() == 0) {
                return mediapipe::tool::StatusStop();
            }

            for (mediapipe::CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId(); ++id) {
                if (!cc->Inputs().Get(id).IsEmpty()) {
                    const mediapipe::InputStreamShard& input_stream = cc->Inputs().Get(id);
                    const std::string& input_stream_name = input_stream.Name();
                    const mediapipe::Packet& packet = input_stream.Value();

                    mediapipe::OutputStreamShard& output_stream = cc->Outputs().Get(id);
                    const std::string& output_stream_name = cc->Outputs().Get(id).Name();

                    std::string&& timestamp_str = cc->InputTimestamp().DebugString();

                    std::cout << "Passing " << input_stream_name << " to " << output_stream_name << " at " << timestamp_str << std::endl;

                    output_stream.AddPacket(packet);
                }
            }

            return absl::OkStatus();
        }
};

REGISTER_CALCULATOR(CustomCalculator);

absl::Status Main() {
    std::cout << "All registered calculators:" << std::endl;
    std::unordered_set<std::string> calculator_names = mediapipe::CalculatorBaseRegistry::GetRegisteredNames();
    for (const std::string& calculator_name : calculator_names) {
        std::cout << "  - " << calculator_name << std::endl;
    }

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        input_stream: "input_stream"
        output_stream: "output_stream"
        node {
            calculator: "CustomCalculator"
            input_stream: "input_stream"
            output_stream: "output_stream"
        }
    )pb");

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));
    graph.ObserveOutputStream("output_stream", [](const mediapipe::Packet& packet) {
        std::cout << "Output packet: " << packet.Get<int>() << std::endl;
        return absl::OkStatus();
    });

    MP_RETURN_IF_ERROR(graph.StartRun({}));
    for (int i = 0; i < 10; i++) {
        std::cout << "Adding packet: " << i << std::endl;
        mediapipe::Packet packet = mediapipe::MakePacket<int>(i).At(mediapipe::Timestamp(i));
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream("input_stream", packet));
    }
    MP_RETURN_IF_ERROR(graph.CloseInputStream("input_stream"));
    MP_RETURN_IF_ERROR(graph.WaitUntilDone());
    return absl::OkStatus();
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);

    absl::Status status = Main();
    if (!status.ok()) {
        std::cerr << "Error: " << status.message() << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Success!" << std::endl;

    return EXIT_SUCCESS;
}

