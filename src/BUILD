cc_binary(
    name = "hello_world",
    srcs = ["hello_world.cc"],
    deps = [
        "@mediapipe//mediapipe/calculators/core:pass_through_calculator",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:logging",
    ],
)

cc_binary(
    name = "face_mesh",
    srcs = ["face_mesh.cc"],
    deps = [
        "@mediapipe//mediapipe/graphs/face_mesh:desktop_calculators",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:file_helpers",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/util:resource_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_binary(
    name = "face_mesh_async",
    srcs = ["face_mesh_async.cc"],
    deps = [
        "@mediapipe//mediapipe/graphs/face_mesh:desktop_calculators",
        "@mediapipe//mediapipe/calculators/image:color_convert_calculator",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:file_helpers",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/util:resource_util",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
)

