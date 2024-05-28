#include <iostream>
#include <filesystem>

#include "absl/status/status.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

#include <SDL2/SDL.h>

ABSL_FLAG(std::string, png_path, "", "Path to a PNG file to show");

const int window_width = 800;

absl::Status RunRenderer(SDL_Window* window, const std::filesystem::path& png_path) {
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

    cv::Mat image = cv::imread(png_path.string());
    if (image.empty()) {
        return absl::InternalError("Failed to load image");
    }

    double aspect_ratio = static_cast<double>(image.cols) / static_cast<double>(image.rows);
    SDL_SetWindowSize(window, window_width, static_cast<int>(static_cast<double>(window_width) / aspect_ratio));

    std::unique_ptr<SDL_Surface, decltype(&SDL_FreeSurface)> surface(
            SDL_CreateRGBSurfaceFrom(
                image.data,
                image.cols,
                image.rows,
                image.elemSize() * 8,
                image.step,
                0xff0000, 0x00ff00, 0x0000ff, 0
                ),
            SDL_FreeSurface
            );
    if (surface == nullptr) {
        std::string error(SDL_GetError());
        return absl::InternalError(error);
    }

    std::unique_ptr<SDL_Texture, decltype(&SDL_DestroyTexture)> texture(
            SDL_CreateTextureFromSurface(renderer.get(), surface.get()),
            SDL_DestroyTexture
            );

    SDL_RenderClear(renderer.get());
    SDL_RenderCopy(renderer.get(), texture.get(), nullptr, nullptr);
    SDL_RenderPresent(renderer.get());

    SDL_Event e;
    bool quit = false;
    while (!quit){
        while (SDL_PollEvent(&e)){
            if (e.type == SDL_QUIT){
                std::cout << "Quit event" << std::endl;
                quit = true;
            }
            if (e.type == SDL_KEYDOWN){
                std::cout << "Key down event" << std::endl;
                quit = true;
            }
            if (e.type == SDL_MOUSEBUTTONDOWN){
                std::cout << "Mouse button down event" << std::endl;
                quit = true;
            }
        }
    }

    return absl::OkStatus();
}

absl::Status RunWindow(const std::filesystem::path& png_path) {
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

    return RunRenderer(window.get(), png_path);
}

absl::Status RunApp(const std::filesystem::path& png_path) {
    std::cout << "Hello, SDL2!" << std::endl;
    SDL_Init(SDL_INIT_VIDEO);

    absl::Status status = RunWindow(std::move(png_path));

    SDL_Quit();
    return status;
}

int main(int argc, char* argv[]) {
    absl::ParseCommandLine(argc, argv);
    std::string png_path = absl::GetFlag(FLAGS_png_path);
    if (png_path.empty()) {
        std::cerr << "Error: --png_path is required" << std::endl;
        return 1;
    }
    std::filesystem::path path(png_path);

    absl::Status status = RunApp(std::move(path));
    if (!status.ok()) {
        std::cerr << "Error: " << status.message() << std::endl;
        return 1;
    }

    return 0;
}

