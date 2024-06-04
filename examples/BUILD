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

cc_binary(
    name = "crop_human",
    srcs = ["crop_human.cc"],
    deps = [
        "//calculators:calculators",

        "@mediapipe//mediapipe/framework:calculator_framework",

        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",

        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",

        "@mediapipe//mediapipe/modules/pose_detection:pose_detection_cpu",

        "@mediapipe//mediapipe/calculators/core:packet_resampler_calculator",
        "@mediapipe//mediapipe/calculators/core:split_vector_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_thinner_calculator",

        "@mediapipe//mediapipe/calculators/util:detections_to_rects_calculator",
        "@mediapipe//mediapipe/calculators/util:detections_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:detection_to_landmarks_calculator",
        "@mediapipe//mediapipe/calculators/util:landmarks_to_detection_calculator",
        "@mediapipe//mediapipe/calculators/util:landmarks_smoothing_calculator",
        "@mediapipe//mediapipe/calculators/util:rect_to_render_data_calculator",
        "@mediapipe//mediapipe/calculators/util:annotation_overlay_calculator",
        "@mediapipe//mediapipe/calculators/util:alignment_points_to_rects_calculator",
        "@mediapipe//mediapipe/calculators/util:rect_transformation_calculator",

        "@mediapipe//mediapipe/calculators/image:image_properties_calculator",
        "@mediapipe//mediapipe/calculators/image:image_cropping_calculator",
        "@mediapipe//mediapipe/calculators/image:scale_image_calculator",

        "@mediapipe//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "@mediapipe//mediapipe/calculators/video:opencv_video_encoder_calculator",
        "@mediapipe//mediapipe/calculators/video:video_pre_stream_calculator",


        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_binary(
    name = "use_custom_calculator",
    srcs = ["use_custom_calculator.cc"],
    deps = [
        "//calculators:calculators",

        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",

        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_binary(
    name = "image_classification",
    srcs = ["image_classification.cc"],
    deps = [
        "//calculators:calculators",

        "@mediapipe//mediapipe/tasks/cc/vision/image_classifier:image_classifier",

        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",

        "@mediapipe//mediapipe/calculators/core:packet_thinner_calculator",

        "@mediapipe//mediapipe/calculators/util:to_image_calculator",

        "@mediapipe//mediapipe/calculators/video:opencv_video_decoder_calculator",

        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
    ],
)

cc_binary(
    name = "hello_inja",
    srcs = ["hello_inja.cc"],
    deps = [
        "@inja//:inja",
    ],
)

cc_binary(
    name = "hello_sdl2",
    srcs = ["hello_sdl2.cc"],
    deps = [
        "//:sdl2",
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
    ],
)

cc_binary(
    name = "video_size_transformation",
    srcs = ["video_size_transformation.cc"],
    deps = [
        "//calculators:calculators",
        "//:sdl2",
        "@inja//:inja",
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_thinner_calculator",
        "@mediapipe//mediapipe/calculators/util:to_image_calculator",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/tasks/cc/vision/image_classifier:image_classifier",
    ],
)

cc_binary(
    name = "simple_tflite_interpreter",
    srcs = ["simple_tflite_interpreter.cc"],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:absl_check",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:type_to_tflitetype",
        "@org_tensorflow//tensorflow/lite:stateful_error_reporter",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels/internal:tensor",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:logging",
    ]
)

cc_binary(
    name = "dataset_from_other_models",
    srcs = ["dataset_from_other_models.cc"],
    deps = [
        "//calculators:calculators",
        "@inja//:inja",
        "@sqlite//:sqlite3",
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@mediapipe//mediapipe/framework:calculator_framework",
        "@mediapipe//mediapipe/framework/formats:image_frame",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/calculators/image:image_transformation_calculator",
        "@mediapipe//mediapipe/calculators/core:packet_thinner_calculator",
        "@mediapipe//mediapipe/calculators/util:to_image_calculator",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
        "@mediapipe//mediapipe/framework/port:opencv_video",
        "@mediapipe//mediapipe/framework/port:parse_text_proto",
        "@mediapipe//mediapipe/tasks/cc/vision/image_classifier:image_classifier",
        "@mediapipe//mediapipe/tasks/cc/vision/face_detector:face_detector",
        "@mediapipe//mediapipe/calculators/image:image_cropping_calculator",
        "@mediapipe//mediapipe/calculators/core:split_vector_calculator",
    ]
)

cc_binary(
    name = "hello_faiss",
    srcs = ["hello_faiss.cc"],
    deps = [
        "//:faiss",
    ],
    linkopts = [
        # XXX I've gave up to build these libraries with Bazel.
        # I think it works only my environment.
        "-L/usr/local/opt/libomp/lib",
        "-L/usr/local/opt/openblas/lib",
        "-lomp",
        "-lopenblas",
    ],
)

cc_binary(
    name="image_similarity",
    srcs=["image_similarity.cc"],
    deps=[
        "@com_google_absl//absl/status:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@mediapipe//mediapipe/util/frame_buffer:frame_buffer_util",
        "@mediapipe//mediapipe/framework/port:status",
        "@mediapipe//mediapipe/tasks/cc/vision/image_embedder:image_embedder",
        "@mediapipe//mediapipe/framework/formats:image_frame_opencv",
        "@mediapipe//mediapipe/framework/port:file_helpers",
        "@mediapipe//mediapipe/framework/port:opencv_highgui",
        "@mediapipe//mediapipe/framework/port:opencv_imgproc",
    ],
)

