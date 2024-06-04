#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <filesystem>

#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "glog/logging.h"

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"

#include <inja/inja.hpp>
#include <SDL2/SDL.h>

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

int window_width = 800;

ABSL_FLAG(std::string, video_path, "", "video path");
ABSL_FLAG(std::string, model_path, "", "model path");

absl::Status RunMediapipe(
        const std::filesystem::path& video_path, 
        const std::filesystem::path& model_path,
        std::queue<std::pair<std::chrono::microseconds, std::unique_ptr<cv::Mat>>>& frame_queue, 
        std::mutex& frame_queue_mutex
        ) {

    inja::Environment env;
    inja::Template calculator_graph_template = env.parse(R"pb(
        output_stream: "output_stream"

        # Input video
        node {
            calculator: "VideoDecoderCalculator"
            input_side_packet: "INPUT_FILE_PATH:input_video_path"
            output_stream: "VIDEO:input_stream"
        }

        node {
            calculator: "PacketThinnerCalculator"
            input_stream: "input_stream"
            output_stream: "downsampled_input_image"
            node_options: {
                [type.googleapis.com/mediapipe.PacketThinnerCalculatorOptions] {
                    thinner_type: ASYNC
                    period: 100000
                }
            }
        }

        node: {
            calculator: "ImageTransformationCalculator"
            input_stream: "IMAGE:downsampled_input_image"
            output_stream: "IMAGE:square_input_image"
            node_options: {
                [type.googleapis.com/mediapipe.ImageTransformationCalculatorOptions] {
                    output_width: 300
                    output_height: 300
                    scale_mode: FIT
                }
            }
        }

        # ImageFrame -> Image (just type conversion)
        node {
            calculator: "ToImageCalculator"
            input_stream: "IMAGE:square_input_image"
            output_stream: "IMAGE:used_by_classifier_image"
        }

        node {
            calculator: "mediapipe.tasks.vision.image_classifier.ImageClassifierGraph"
            input_stream: "IMAGE:used_by_classifier_image"
            output_stream: "CLASSIFICATIONS:classifications"
            output_stream: "IMAGE:classified_image"
            node_options {
                [type.googleapis.com/mediapipe.tasks.vision.image_classifier.proto.ImageClassifierGraphOptions] {
                    base_options {
                        model_asset {
                            file_name: "{{ model_path }}" # TODO escape path
                        }
                    }
                    classifier_options {
                        score_threshold: 0.6
                    }
                }
            }
        }
        node {
            calculator: "ClassificationImageGateCalculator"
            input_stream: "CLASSIFICATIONS:classifications"
            input_stream: "IMAGE:classified_image"
            output_stream: "LABEL_SCORE_IMAGE:output_stream"
        }
    )pb");

    std::string calculator_graph_str = env.render(calculator_graph_template, {{"model_path", model_path.string()}});
    LOG(INFO) << "Calculator graph: " << calculator_graph_str;
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_str);

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    MP_RETURN_IF_ERROR(graph.ObserveOutputStream("output_stream", [&frame_queue, &frame_queue_mutex](const mediapipe::Packet& packet) {
                const mediapipe::Timestamp& timestamp = packet.Timestamp();
                const std::chrono::microseconds timestamp_us = std::chrono::microseconds(timestamp.Microseconds());
                LOG(INFO) << "Received frame at " << timestamp.Microseconds();

                std::pair<std::vector<std::pair<std::string, double>>, std::shared_ptr<const mediapipe::ImageFrame>> label_score_frame = packet.Get<std::pair<std::vector<std::pair<std::string, double>>, std::shared_ptr<const mediapipe::ImageFrame>>>();
                std::vector<std::pair<std::string, double>> label_score = label_score_frame.first;
                std::shared_ptr<const mediapipe::ImageFrame> frame = label_score_frame.second;

                // TODO no more effective, video decoder calculator must make a boolean flag stream only for the last frame
                if (frame->IsEmpty()) {
                    std::unique_ptr<cv::Mat> end_of_video(nullptr);
                    {
                        std::lock_guard<std::mutex> lock(frame_queue_mutex);
                        frame_queue.push({ timestamp_us, std::move(end_of_video) });
                    }
                    return absl::OkStatus();
                }

                // ensure the frame copy here
                std::unique_ptr<cv::Mat> frame_mat(new cv::Mat(frame->Height(), frame->Width(), CV_8UC3));
                if (frame_mat->channels() == 3) {
                    cv::cvtColor(mediapipe::formats::MatView(&*frame), *frame_mat, cv::COLOR_RGB2BGR);
                } else {
                    return absl::InternalError("Unsupported number of channels");
                }
                // rough serialize the label and score
                std::string label_score_str;
                for (const auto& label_score_pair : label_score) {
                    label_score_str += label_score_pair.first + ": " + std::to_string(label_score_pair.second) + "\n";
                }
                ABSL_ASSERT(frame_mat->data != nullptr);
                cv::putText(*frame_mat, label_score_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

                {
                    std::lock_guard<std::mutex> lock(frame_queue_mutex);
                    frame_queue.push({ timestamp_us, std::move(frame_mat) });
                }
                return absl::OkStatus();
                }));

    MP_RETURN_IF_ERROR(graph.StartRun({
                { "input_video_path", mediapipe::MakePacket<std::string>(video_path.string()) }
                }));

    MP_RETURN_IF_ERROR(graph.WaitUntilDone());

    return absl::OkStatus();
}

absl::Status RunPlayer(
        SDL_Window* window, 
        SDL_Renderer& renderer, 
        std::queue<std::pair<std::chrono::microseconds, std::unique_ptr<cv::Mat>>>& frame_queue, 
        std::mutex& frame_queue_mutex
        ) {
    std::unordered_map<std::pair<int, int>, std::shared_ptr<SDL_Texture>, pair_hash> texture_cache;
    std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();

    SDL_RenderClear(&renderer);
    SDL_SetRenderDrawColor(&renderer, 0, 0, 0, 255);
    SDL_RenderPresent(&renderer);

    bool done = false;
    while (!done) {
        cv::Mat *frame_to_show = nullptr;
        {
            std::unique_lock<std::mutex> lock(frame_queue_mutex);

            while (!frame_queue.empty()) {
                std::unique_ptr<cv::Mat> frame_candidate = std::move(frame_queue.front().second);
                frame_queue.pop();

                if (frame_candidate == nullptr) {
                    // this indicates the end of the video
                    done = true;
                    break;
                }
                frame_to_show = frame_candidate.release();
            }
        }
        if (done) {
            break;
        }
        if (frame_to_show != nullptr) {
            std::unique_ptr<cv::Mat> frame(frame_to_show);
            std::unique_ptr<SDL_Surface, decltype(&SDL_FreeSurface)> surface(
                    SDL_CreateRGBSurfaceFrom(
                        frame->data,
                        frame->cols,
                        frame->rows,
                        frame->elemSize() * 8,
                        frame->step,
                        0xff0000, 0x00ff00, 0x0000ff, 0
                        ),
            SDL_FreeSurface
            );
            if (surface == nullptr) {
                std::string error(SDL_GetError());
                return absl::InternalError(error);
            }

            int surface_width = surface->w;
            int surface_height = surface->h;

            double aspect_ratio = static_cast<double>(surface_width) / static_cast<double>(surface_height);
            SDL_SetWindowSize(window, window_width, static_cast<int>(static_cast<double>(window_width) / aspect_ratio));

            std::pair<int, int> surface_size(surface_width, surface_height);
            auto it = texture_cache.find(surface_size);
            if (it == texture_cache.end()) {
                std::shared_ptr<SDL_Texture> texture(
                        SDL_CreateTexture(
                            &renderer,
                            SDL_PIXELFORMAT_BGR24,
                            SDL_TEXTUREACCESS_STREAMING,
                            surface_width,
                            surface_height
                            ),
                        SDL_DestroyTexture
                        );
                if (texture == nullptr) {
                    std::string error(SDL_GetError());
                    return absl::InternalError(error);
                }
                texture_cache[surface_size] = texture;
            }
            SDL_Texture* texture = texture_cache[surface_size].get();
            ABSL_ASSERT(texture != nullptr);

            SDL_UpdateTexture(texture, nullptr, surface->pixels, surface->pitch);

            SDL_RenderClear(&renderer);
            SDL_RenderCopy(&renderer, texture, nullptr, nullptr);
            SDL_RenderPresent(&renderer);
        }

        SDL_Event e;
        while (SDL_PollEvent(&e)){
            if (e.type == SDL_QUIT){
                std::cout << "Quit event" << std::endl;
                return absl::OkStatus();
            }
            if (e.type == SDL_KEYDOWN){
                std::cout << "Key down event" << std::endl;
                return absl::OkStatus();
            }
            if (e.type == SDL_MOUSEBUTTONDOWN){
                std::cout << "Mouse button down event" << std::endl;
                return absl::OkStatus();
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return absl::OkStatus();
}

int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);

    absl::ParseCommandLine(argc, argv);
    std::string video_path_str = absl::GetFlag(FLAGS_video_path);
    if (video_path_str.empty()) {
        LOG(ERROR) << "Please specify video path with --video_path flag";
        return EXIT_FAILURE;
    }
    std::filesystem::path video_path(video_path_str);

    std::string model_path_str = absl::GetFlag(FLAGS_model_path);
    if (model_path_str.empty()) {
        LOG(ERROR) << "Please specify model path with --model_path flag";
        return EXIT_FAILURE;
    }
    std::filesystem::path model_path(model_path_str);

    // set of frame queue
    std::queue<std::pair<std::chrono::microseconds, std::unique_ptr<cv::Mat>>> frame_queue;
    std::mutex frame_queue_mutex;

    absl::Status mediapipe_status;
    std::thread mediapipe_thread([&frame_queue, &frame_queue_mutex, &mediapipe_status, &video_path, &model_path]() {
        mediapipe_status = RunMediapipe(video_path, model_path, frame_queue, frame_queue_mutex);
        LOG(INFO) << "Mediapipe thread finished: " << mediapipe_status.message();
    });

    std::string file_stem = video_path.stem().string();
    absl::Status player_status;
    SDL_Init(SDL_INIT_VIDEO);
    {
        std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
                SDL_CreateWindow(
                    file_stem.c_str(),
                    SDL_WINDOWPOS_CENTERED,
                    SDL_WINDOWPOS_CENTERED,
                    window_width,
                    window_width,
                    SDL_WINDOW_SHOWN
                    ),
                SDL_DestroyWindow
                );
        if (window == nullptr) {
            std::string error(SDL_GetError());
            player_status = absl::InternalError(error);
        } else {
            SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");
            std::unique_ptr<SDL_Renderer, decltype(&SDL_DestroyRenderer)> renderer(
                    SDL_CreateRenderer(window.get(), -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC),
                    SDL_DestroyRenderer
                    );
            if (renderer == nullptr) {
                std::string error(SDL_GetError());
                player_status = absl::InternalError(error);
            }
            SDL_RendererInfo renderer_info;
            SDL_GetRendererInfo(renderer.get(), &renderer_info);
            LOG(INFO) << "Renderer: " << renderer_info.name;
            player_status = RunPlayer(window.get(), *renderer, frame_queue, frame_queue_mutex);
        }
    }
    SDL_Quit();

    mediapipe_thread.join();

    if (!mediapipe_status.ok() || !player_status.ok()) {
        LOG(ERROR) << "Error: " << "(Mediapipe: " << mediapipe_status.message() << ", Player: " << player_status.message() << ")";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

