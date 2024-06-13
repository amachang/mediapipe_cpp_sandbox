package(default_visibility = ["//visibility:public"])

cc_library(
    name = "utils",
    deps = [
        ":recursive_dir_iter",
        ":image_database",
    ],
)

cc_library(
    name = "image_database",
    srcs = ["image_database.cc"],
    hdrs = ["image_database.h"],
    deps = [
        "//:faiss",
        "@sqlite//:sqlite3",
        "@mediapipe//mediapipe/util/frame_buffer:frame_buffer_util",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/tasks/cc/vision/image_embedder:image_embedder",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:file_helpers",
    ],
)

cc_library(
    name = "recursive_dir_iter",
    hdrs = ["recursive_dir_iter.h"],
)

