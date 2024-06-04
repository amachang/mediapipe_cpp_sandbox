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
#       "//examples:hello_sdl2": "",
#       "//calculators:calculators": "",
#       "//examples:video_size_transformation": "",
#       "//examples:simple_tflite_interpreter": "",
#       "//examples:dataset_from_other_models": "",
        "//examples:hello_faiss": "",
        "//examples:image_similarity": "",
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

cmake(
    name = "faiss",
    lib_source = "@faiss//:all",
    out_static_libs = ["libfaiss.a"],
    visibility = ["//visibility:public"],
    generate_args = [
        "-G Ninja",
        "-DFAISS_ENABLE_GPU=OFF",
        "-DFAISS_ENABLE_PYTHON=OFF",
        # disable test
        "-DBUILD_TESTING=OFF",

        # XXX it only works with my env
        "-DOpenMP_CXX_FLAGS=\"-Xpreprocessor -fopenmp -I/usr/local/opt/libomp/include\"",
        "-DOpenMP_CXX_LIB_NAMES=omp",
        "-DOpenMP_omp_LIBRARY=/usr/local/opt/libomp/lib/libomp.dylib",
    ],
)

