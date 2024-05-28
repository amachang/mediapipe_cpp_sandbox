#include "absl/status/status.h"
#include <iostream>
#include <SDL2/SDL.h>

absl::Status RunRenderer(SDL_Window* window) {
    std::unique_ptr<SDL_Renderer, decltype(&SDL_DestroyRenderer)> renderer = std::unique_ptr<SDL_Renderer, decltype(&SDL_DestroyRenderer)>(SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED), SDL_DestroyRenderer);
    if (renderer == nullptr) {
        std::string error(SDL_GetError());
        return absl::InternalError(error);
    }

    SDL_SetRenderDrawColor(renderer.get(), 96, 128, 255, 255);
    SDL_RenderClear(renderer.get());
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

absl::Status RunWindow() {
    std::unique_ptr<SDL_Window, decltype(&SDL_DestroyWindow)> window(
            SDL_CreateWindow("Hello, SDL2!", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, SDL_WINDOW_SHOWN),
            SDL_DestroyWindow
        );
    if (window == nullptr) {
        std::string error(SDL_GetError());
        return absl::InternalError(error);
    }

    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");

    return RunRenderer(window.get());
}

absl::Status RunApp() {
    std::cout << "Hello, SDL2!" << std::endl;
    SDL_Init(SDL_INIT_VIDEO);

    absl::Status status = RunWindow();

    SDL_Quit();
    return status;
}

int main() {
    absl::Status status = RunApp();
    if (!status.ok()) {
        std::cerr << "Error: " << status.message() << std::endl;
        return 1;
    }
    return 0;
}

