load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "calculators",
    deps = [
        ":detection_to_largest_square_rect_calculator",
        ":classification_image_gate_calculator",
        ":video_decoder_calculator",
    ],
)

cc_library(
    name = "detection_to_largest_square_rect_calculator",
    srcs = ["detection_to_largest_square_rect_calculator.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        ":detection_to_largest_square_rect_calculator_cc_proto",
        "@mediapipe//mediapipe/framework/formats:rect_cc_proto",
        "@mediapipe//mediapipe/framework/formats:detection_cc_proto",
    ],
    alwayslink = 1,
)

mediapipe_proto_library(
    name = "detection_to_largest_square_rect_calculator_proto",
    srcs = ["detection_to_largest_square_rect_calculator.proto"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_proto",
    ],
)

cc_library(
    name = "classification_image_gate_calculator",
    srcs = ["classification_image_gate_calculator.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/tasks/cc/components/containers:classification_result",
    ],
    alwayslink = 1,
)

cc_library(
    name = "video_decoder_calculator",
    srcs = ["video_decoder_calculator.cc"],
    deps = [
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image_format_cc_proto",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/formats:video_stream_header",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/framework/tool:status_util",
        "@com_google_absl//absl/log:absl_log",
    ],
    alwayslink = 1,
)
