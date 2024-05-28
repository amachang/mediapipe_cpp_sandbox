load("@hedron_compile_commands//:refresh_compile_commands.bzl", "refresh_compile_commands")

refresh_compile_commands(
    name = "refresh_compile_commands",

    targets = {
#       "//examples:hello_world": "",
#       "//examples:face_mesh": "",
#       "//examples:face_mesh_async": "",
#       "//examples:crop_human": "",
#       "//examples:use_custom_calculator": "",
#       "//examples:image_classification": "",
#       "//examples:hello_inja": "",
      "//examples:hello_sdl2": "",
#       "//calculators:calculators": "",
    },
)

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

cmake(
    name = "sdl2",
    lib_source = "@sdl2//:all",
    out_static_libs = ["libsdl2.a"],
    visibility = ["//visibility:public"],
    deps = [":sdl2_apple_dependencies"],
    linkopts = [
        "-liconv",
    ],
)

cc_library(
    name = "sdl2_apple_dependencies",
    linkopts = [
        "-framework GameController",
        "-framework ForceFeedback",
        "-framework CoreHaptics",
        "-framework CoreFoundation",
        "-framework CoreServices",
        "-framework Cocoa",
        "-framework CoreAudio",
        "-framework CoreVideo",
        "-framework AudioToolbox",
        "-framework VideoToolbox",
        "-framework IOKit",
        "-framework Carbon",
        "-framework Metal",
        "-framework AppKit",
    ],
)

