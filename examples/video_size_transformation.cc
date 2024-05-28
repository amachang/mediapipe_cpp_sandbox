#include "absl/flags/parse.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "glog/logging.h"
#include <thread>
#include <SDL2/SDL.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <filesystem>

#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/tasks/cc/components/containers/classification_result.h"

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

int target_size = 800;

ABSL_FLAG(std::string, video_path, "", "video path");

absl::Status RunMediapipe(
        const std::filesystem::path& video_path, 
        std::queue<std::pair<std::chrono::microseconds, std::unique_ptr<cv::Mat>>>& frame_queue, 
        std::mutex& frame_queue_mutex
        ) {

    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(R"pb(
        output_stream: "output_stream"

        # Input video
        node {
            calculator: "VideoDecoderCalculator"
            input_side_packet: "INPUT_FILE_PATH:input_video_path"
            output_stream: "VIDEO:output_stream"
        }
    )pb");

    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    MP_RETURN_IF_ERROR(graph.ObserveOutputStream("output_stream", [&frame_queue, &frame_queue_mutex](const mediapipe::Packet& packet) {
                const mediapipe::Timestamp& timestamp = packet.Timestamp();
                LOG(INFO) << "Timestamp: " << timestamp.Microseconds();
                const std::chrono::microseconds timestamp_us = std::chrono::microseconds(timestamp.Microseconds());
                const mediapipe::ImageFrame& frame = packet.Get<mediapipe::ImageFrame>();
                if (frame.IsEmpty()) {
                    std::unique_ptr<cv::Mat> end_of_video(nullptr);
                    {
                        std::lock_guard<std::mutex> lock(frame_queue_mutex);
                        frame_queue.push({ timestamp_us, std::move(end_of_video) });
                    }
                    return absl::OkStatus();
                }

                // ensure the frame copy here
                cv::Mat frame_mat(frame.Height(), frame.Width(), CV_8UC3);
                if (frame_mat.channels() == 3) {
                    cv::cvtColor(mediapipe::formats::MatView(&frame), frame_mat, cv::COLOR_RGB2BGR);
                } else {
                    return absl::InternalError("Unsupported number of channels");
                }
                ABSL_ASSERT(frame_mat.data != nullptr);

                // resize and pad, keeping aspect ratio
                int frame_width = frame_mat.cols;
                int frame_height = frame_mat.rows;
                double aspect_ratio = static_cast<double>(frame_width) / static_cast<double>(frame_height);
                std::unique_ptr<cv::Mat> frame_mat_resized = std::make_unique<cv::Mat>(target_size, target_size, CV_8UC3, cv::Scalar(0, 0, 0));
                // put the resized frame in the center
                if (aspect_ratio > 1.0) {
                    int resized_height = static_cast<int>(static_cast<double>(target_size) / aspect_ratio);
                    cv::Mat resized_frame(*frame_mat_resized, cv::Rect(0, (target_size - resized_height) / 2, target_size, resized_height));
                    cv::resize(frame_mat, resized_frame, resized_frame.size(), 0, 0, cv::INTER_LINEAR);
                } else {
                    int resized_width = static_cast<int>(static_cast<double>(target_size) * aspect_ratio);
                    cv::Mat resized_frame(*frame_mat_resized, cv::Rect((target_size - resized_width) / 2, 0, resized_width, target_size));
                    cv::resize(frame_mat, resized_frame, resized_frame.size(), 0, 0, cv::INTER_LINEAR);
                }

                {
                    std::lock_guard<std::mutex> lock(frame_queue_mutex);
                    frame_queue.push({ timestamp_us, std::move(frame_mat_resized) });
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
            SDL_SetWindowSize(window, target_size, static_cast<int>(static_cast<double>(target_size) / aspect_ratio));

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

    // set of frame queue
    std::queue<std::pair<std::chrono::microseconds, std::unique_ptr<cv::Mat>>> frame_queue;
    std::mutex frame_queue_mutex;

    absl::Status mediapipe_status;
    std::thread mediapipe_thread([&frame_queue, &frame_queue_mutex, &mediapipe_status, &video_path]() {
        mediapipe_status = RunMediapipe(video_path, frame_queue, frame_queue_mutex);
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
                    target_size,
                    target_size,
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

