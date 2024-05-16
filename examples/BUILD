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

