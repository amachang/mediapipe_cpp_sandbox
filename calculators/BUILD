load("@mediapipe//mediapipe/framework/port:build_config.bzl", "mediapipe_proto_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "calculators",
    deps = [
        ":detection_to_largest_square_rect_calculator",
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

