#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <thread>
#include <chrono>

#include "absl/status/status.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

#include <SDL2/SDL.h>

ABSL_FLAG(std::string, video_path, "", "Path to a Video file to show");

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

const int window_width = 800;

absl::Status RunRenderer(SDL_Window* window, const std::filesystem::path& video_path) {
    std::unique_ptr<SDL_Renderer, decltype(&SDL_DestroyRenderer)> renderer(
            SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC),
            SDL_DestroyRenderer
            );
    if (renderer == nullptr) {
        std::string error(SDL_GetError());
        return absl::InternalError(error);
    }

    SDL_RendererInfo renderer_info;
    SDL_GetRendererInfo(renderer.get(), &renderer_info);
    std::cout << "Renderer: " << renderer_info.name << std::endl;

    std::unordered_map<std::pair<int, int>, std::shared_ptr<SDL_Texture>, pair_hash> texture_cache;

    cv::VideoCapture cap(video_path.string());

    // here is the 0 msec
    std::chrono::milliseconds start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());

    int frame_index = 0;
    while (cap.isOpened()) {
        double frame_time = cap.get(cv::CAP_PROP_POS_MSEC);
        std::chrono::milliseconds current_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        std::chrono::milliseconds elapsed_time = current_time - start_time;
        if (elapsed_time.count() < frame_time) {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(frame_time - elapsed_time.count())));
        }

        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        int frame_width = frame.cols;
        int frame_height = frame.rows;

        double aspect_ratio = static_cast<double>(frame_width) / static_cast<double>(frame_height);
        SDL_SetWindowSize(window, window_width, static_cast<int>(static_cast<double>(window_width) / aspect_ratio));

        std::pair<int, int> frame_size(frame.cols, frame.rows);
        auto it = texture_cache.find(frame_size);
        if (it == texture_cache.end()) {
            std::shared_ptr<SDL_Texture> texture(
                    SDL_CreateTexture(
                        renderer.get(),
                        SDL_PIXELFORMAT_BGR24,
                        SDL_TEXTUREACCESS_STREAMING,
                        frame_size.first,
                        frame_size.second
                        ),
                    SDL_DestroyTexture
                    );
            if (texture == nullptr) {
                std::string error(SDL_GetError());
                return absl::InternalError(error);
            }
            texture_cache[frame_size] = texture;
        }
        SDL_Texture* texture = texture_cache[frame_size].get();
        ABSL_ASSERT(texture != nullptr);

        std::unique_ptr<SDL_Surface, decltype(&SDL_FreeSurface)> surface(
                SDL_CreateRGBSurfaceFrom(
                    frame.data,
                    frame.cols,
                    frame.rows,
                    frame.elemSize() * 8,
                    frame.step,
                    0xff0000, 0x00ff00, 0x0000ff, 0
                    ),
                SDL_FreeSurface
                );
        if (surface == nullptr) {
            std::string error(SDL_GetError());
            return absl::InternalError(error);
        }
        
        SDL_UpdateTexture(texture, nullptr, surface->pixels, surface->pitch);

        SDL_RenderClear(renderer.get());
        SDL_RenderCopy(renderer.get(), texture, nullptr, nullptr);
        SDL_RenderPresent(renderer.get());

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
    }
    return absl::OkStatus();
}

absl::Status RunWindow(const std::filesystem::path& video_path) {
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
            SDL_CreateWindow(
                "Hello, SDL2!",
                SDL_WINDOWPOS_CENTERED,
                SDL_WINDOWPOS_CENTERED,
                window_width,
                window_width * 3 / 4,
                SDL_WINDOW_SHOWN
                ),
            SDL_DestroyWindow
            );
    if (window == nullptr) {
        std::string error(SDL_GetError());
        return absl::InternalError(error);
    }

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");

    return RunRenderer(window.get(), video_path);
}

absl::Status RunApp(const std::filesystem::path& video_path) {
    std::cout << "Hello, SDL2!" << std::endl;
    SDL_Init(SDL_INIT_VIDEO);

    absl::Status status = RunWindow(std::move(video_path));

    SDL_Quit();
    return status;
}

int main(int argc, char* argv[]) {
    absl::ParseCommandLine(argc, argv);
    std::string video_path = absl::GetFlag(FLAGS_video_path);
    if (video_path.empty()) {
        std::cerr << "Error: --video_path is required" << std::endl;
        return 1;
    }
    std::filesystem::path path(video_path);

    absl::Status status = RunApp(std::move(path));
    if (!status.ok()) {
        std::cerr << "Error: " << status.message() << std::endl;
        return 1;
    }

    return 0;
}

