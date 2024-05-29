#include <string>
#include <filesystem>

#include "absl/status/status.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "absl/log/absl_check.h"
#include "glog/logging.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/stateful_error_reporter.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"

ABSL_FLAG(std::string, model_path, "", "Path to the TFLite model");
ABSL_FLAG(std::string, image_path, "", "Path to the image");

class ErrorReporter : public tflite::StatefulErrorReporter {
    public:
        ErrorReporter() {
        }

        int Report(const char* format, std::va_list args) override {
            char message_buf[1024];
            int message_len = vsnprintf(message_buf, sizeof(message_buf), format, args);
            if (message_len < 0) {
                return -1;
            }
            std::string message(message_buf, message_len);
            messages_.push_back(message);
            return message_len;
        }

        bool HasError() const {
            return !messages_.empty();
        }

        std::string message() override {
            return messages_.back();
        }

    private:
        std::vector<std::string> messages_;
};

std::string TfLiteTypeToString(TfLiteType type) {
    switch (type) {
        case kTfLiteFloat32: return "FLOAT32";
        case kTfLiteFloat16: return "FLOAT16";
        case kTfLiteInt32: return "INT32";
        case kTfLiteUInt8: return "UINT8";
        case kTfLiteInt64: return "INT64";
        case kTfLiteString: return "STRING";
        case kTfLiteBool: return "BOOL";
        case kTfLiteInt16: return "INT16";
        case kTfLiteComplex64: return "COMPLEX64";
        case kTfLiteInt8: return "INT8";
        case kTfLiteFloat64: return "FLOAT64";
        case kTfLiteComplex128: return "COMPLEX128";
        case kTfLiteUInt64: return "UINT64";
        case kTfLiteResource: return "RESOURCE";
        case kTfLiteVariant: return "VARIANT";
        case kTfLiteUInt32: return "UINT32";
        case kTfLiteUInt16: return "UINT16";
        case kTfLiteInt4: return "INT4";
        case kTfLiteBFloat16: return "BFLOAT16";
        default: return "UNKNOWN";
    }
}

absl::Status DumpTfLiteTensorDetails(const TfLiteTensor& tensor) {
    TfLiteIntArray* dims = tensor.dims;
    std::vector<int> size(dims->data, dims->data + dims->size);
    const TfLiteIntArray* dims_signature = tensor.dims_signature;
    std::optional<std::vector<int>> size_signature(std::nullopt);
    if (dims_signature != nullptr) {
        size_signature = std::vector<int>(dims_signature->data, dims_signature->data + dims_signature->size);
    }
    TfLiteType type = tensor.type;
    TfLiteQuantizationParams quant_params = tensor.params;
    TfLiteQuantization quant = tensor.quantization;
    TfLiteSparsity* sparsity = tensor.sparsity;

    std::cout << "Tensor details:" << std::endl;
    std::cout << "- Tensor name: " << tensor.name << std::endl;
    std::cout << "- Tensor size:" << std::endl;
    for (int i = 0; i < size.size(); i++) {
        std::cout << "  - " << size[i] << std::endl;
    }
    if (size_signature.has_value()) {
        std::cout << "- Tensor size signature:" << std::endl;
        for (int i = 0; i < size_signature->size(); i++) {
            std::cout << "  - " << (*size_signature)[i] << std::endl;
        }
    } else {
        std::cout << "- Tensor size signature: None" << std::endl;
    }
    std::cout << "- Tensor type: " << TfLiteTypeToString(type) << std::endl;
    std::cout << "- Tensor quantization:" << std::endl;
    std::cout << "  - Scale: " << quant_params.scale << std::endl;
    std::cout << "  - Zero point: " << quant_params.zero_point << std::endl;
    if (quant.type == kTfLiteAffineQuantization) {
        std::cout << "- Tensor quantization (multiple version):" << std::endl;
        const TfLiteAffineQuantization* quant_params = reinterpret_cast<const TfLiteAffineQuantization*>(quant.params);
        if (quant_params->scale != nullptr) {
            const TfLiteFloatArray* scale_array = quant_params->scale;
            std::vector<float> scales(scale_array->data, scale_array->data + scale_array->size);
            std::cout << "  - Scales:" << std::endl;
            for (int i = 0; i < scales.size(); i++) {
                std::cout << "    - " << scales[i] << std::endl;
            }
        }
        if (quant_params->zero_point != nullptr) {
            const TfLiteIntArray* zero_point_array = quant_params->zero_point;
            std::vector<int> zero_points(zero_point_array->data, zero_point_array->data + zero_point_array->size);
            std::cout << "  - Zero points:" << std::endl;
            for (int i = 0; i < zero_points.size(); i++) {
                std::cout << "    - " << zero_points[i] << std::endl;
            }
        }
        std::cout << "  - Quantized dimension: " << quant_params->quantized_dimension << std::endl;
    } else {
        ABSL_CHECK(quant.type == kTfLiteNoQuantization);
        std::cout << "- Tensor quantization (multiple version): None" << std::endl;
    }
    if (sparsity != nullptr) {
        std::cout << "- Tensor sparsity:" << std::endl;
        const TfLiteIntArray* traversal_order_array = sparsity->traversal_order;
        std::vector<int> traversal_order(traversal_order_array->data, traversal_order_array->data + traversal_order_array->size);
        std::cout << "  - Traversal order:" << std::endl;
        for (int i = 0; i < traversal_order.size(); i++) {
            std::cout << "  - " << traversal_order[i] << std::endl;
        }
        const TfLiteIntArray* block_map_array = sparsity->block_map;
        std::vector<int> block_map(block_map_array->data, block_map_array->data + block_map_array->size);
        std::cout << "  - Block map:" << std::endl;
        for (int i = 0; i < block_map.size(); i++) {
            std::cout << "  - " << block_map[i] << std::endl;
        }
        int dim_metadata_size = sparsity->dim_metadata_size;
        for (int i = 0; i < dim_metadata_size; i++) {
            const TfLiteDimensionMetadata& metadata = sparsity->dim_metadata[i];
            std::cout << "  - Dimension metadata:" << std::endl;
            if (metadata.format == kTfLiteDimDense) {
                std::cout << "    - Format: Dense(" << metadata.dense_size << ")" << std::endl;
            } else {
                std::cout << "    - Format: Sparse" << std::endl;
                const TfLiteIntArray* segments_array = metadata.array_segments;
                std::vector<int> segments(segments_array->data, segments_array->data + segments_array->size);
                std::cout << "    - Segments:" << std::endl;
                for (int i = 0; i < segments.size(); i++) {
                    std::cout << "    - " << segments[i] << std::endl;
                }
                const TfLiteIntArray* indices_array = metadata.array_indices;
                std::vector<int> indices(indices_array->data, indices_array->data + indices_array->size);
                std::cout << "    - Indices:" << std::endl;
                for (int i = 0; i < indices.size(); i++) {
                    std::cout << "    - " << indices[i] << std::endl;
                }
            }
        }
    } else {
        std::cout << "- Tensor sparsity: None" << std::endl;
    }
    return absl::OkStatus();
}

absl::Status Interpret(const std::filesystem::path& model_path, const std::filesystem::path& image_path) {
    assert(tflite::MMAPAllocation::IsSupported());
    assert(std::filesystem::exists(model_path));
    assert(std::filesystem::exists(image_path));

    ErrorReporter error_reporter;
    std::unique_ptr<tflite::Allocation> allocation = std::make_unique<tflite::MMAPAllocation>(model_path.c_str(), &error_reporter);
    if (error_reporter.HasError()) {
        return absl::InternalError(error_reporter.message());
    }
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromAllocation(std::move(allocation), &error_reporter);
    if (error_reporter.HasError()) {
        return absl::InternalError(error_reporter.message());
    }

    tflite::ops::builtin::BuiltinOpResolver op_resolver;
    tflite::InterpreterBuilder interpreter_builder(*model, op_resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    interpreter_builder(&interpreter);

    ABSL_CHECK(interpreter != nullptr);
    ABSL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    tflite::Subgraph& subgraph = *(interpreter->subgraph(0));

    std::cout << "--- input tensor ---" << std::endl;

    std::vector<int> input_indices = interpreter->inputs();
    if (input_indices.size() != 1) {
        return absl::InternalError("Multiple input tensors are not supported");
    }
    int input_tensor_index = input_indices[0];
    TfLiteTensor& input_tensor = *(subgraph.tensor(input_tensor_index));
    DumpTfLiteTensorDetails(input_tensor);

    std::cout << "--- output tensor ---" << std::endl;

    std::vector<int> output_indices = interpreter->outputs();
    if (output_indices.size() != 1) {
        return absl::InternalError("Multiple output tensors are not supported");
    }

    int output_tensor_index = output_indices[0];
    TfLiteTensor& output_tensor = *(subgraph.tensor(output_tensor_index));
    DumpTfLiteTensorDetails(output_tensor);

    std::cout << "--- check for invocation ---" << std::endl;

    TfLiteIntArray* input_dims = input_tensor.dims;
    std::vector<int> input_size(input_dims->data, input_dims->data + input_dims->size);
    const TfLiteIntArray* input_dims_signature = input_tensor.dims_signature;
    std::optional<std::vector<int>> input_size_signature(std::nullopt);
    TfLiteType input_type = input_tensor.type;
    TfLiteQuantizationParams input_quant_params = input_tensor.params;
    TfLiteQuantization input_quant = input_tensor.quantization;
    TfLiteSparsity* input_sparsity = input_tensor.sparsity;

    if (input_size.size() != 4) {
        return absl::UnimplementedError("Input tensor size is not supported");
    }
    if (input_size[0] != 1) {
        return absl::UnimplementedError("Currently only batch size 1 is supported");
    }
    if (input_size[3] != 3) {
        return absl::UnimplementedError("Currently only 3-channel input is supported");
    }

    if (input_size_signature.has_value()) {
        if (input_size_signature->size() != 4) {
            return absl::UnimplementedError("Input tensor size signature is not supported");
        }
        if (input_size_signature->at(0) != -1) {
            return absl::UnimplementedError("Currently only batch size 1 is supported");
        }
        if (input_size_signature->at(1) != input_size[1]) {
            return absl::UnimplementedError("Input tensor height is not supported");
        }
        if (input_size_signature->at(2) != input_size[2]) {
            return absl::UnimplementedError("Input tensor width is not supported");
        }
        if (input_size_signature->at(3) != 3) {
            return absl::UnimplementedError("Currently only 3-channel input is supported");
        }
    }

    if (input_type != kTfLiteFloat32) {
        return absl::UnimplementedError("Currently only FLOAT32 input type is supported");
    }

    if (input_quant_params.scale != 0.0 || input_quant_params.zero_point != 0) {
        return absl::UnimplementedError("Currently only zero-centered FLOAT32 input is supported");
    }

    if (input_quant.type != kTfLiteNoQuantization) {
        return absl::UnimplementedError("Currently only no quantization input is supported");
    }

    if (input_sparsity != nullptr) {
        return absl::UnimplementedError("Sparsity is not supported");
    }

    TfLiteIntArray* output_dims = output_tensor.dims;
    std::vector<int> output_size(output_dims->data, output_dims->data + output_dims->size);
    const TfLiteIntArray* output_dims_signature = output_tensor.dims_signature;
    std::optional<std::vector<int>> output_size_signature(std::nullopt);
    TfLiteType output_type = output_tensor.type;
    TfLiteQuantizationParams output_quant_params = output_tensor.params;
    TfLiteQuantization output_quant = output_tensor.quantization;
    TfLiteSparsity* output_sparsity = output_tensor.sparsity;

    if (output_size.size() != 2) {
        return absl::UnimplementedError("Output tensor size is not supported");
    }
    if (output_size[0] != 1) {
        return absl::UnimplementedError("Currently only batch size 1 is supported");
    }

    if (output_size_signature.has_value()) {
        if (output_size_signature->size() != 2) {
            return absl::UnimplementedError("Output tensor size signature is not supported");
        }
        if (output_size_signature->at(0) != -1) {
            return absl::UnimplementedError("Currently only batch size 1 is supported");
        }
        if (output_size_signature->at(1) != output_size[1]) {
            return absl::UnimplementedError("Output tensor height is not supported");
        }
    }

    if (output_type != kTfLiteFloat32) {
        return absl::UnimplementedError("Currently only FLOAT32 output type is supported");
    }

    if (output_quant_params.scale != 0.0 || output_quant_params.zero_point != 0) {
        return absl::UnimplementedError("Currently only zero-centered FLOAT32 output is supported");
    }

    if (output_quant.type != kTfLiteNoQuantization) {
        return absl::UnimplementedError("Currently only no quantization output is supported");
    }

    if (output_sparsity != nullptr) {
        return absl::UnimplementedError("Sparsity is not supported");
    }

    std::cout << "Ok" << std::endl;

    std::cout << "--- inference ---" << std::endl;

    cv::Mat image = cv::imread(image_path.string());
    ABSL_CHECK(!image.empty());

    cv::Size resized_size(input_size[2], input_size[1]);
    double width_ratio = static_cast<double>(resized_size.width) / image.cols;
    double height_ratio = static_cast<double>(resized_size.height) / image.rows;
    bool width_longer = width_ratio > height_ratio;
    double scale = width_longer ? height_ratio : width_ratio;
    cv::Size new_size(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale));
    cv::Mat resized_image(new_size, image.type());
    cv::resize(image, resized_image, new_size, 0, 0, cv::INTER_LINEAR);
    cv::Mat padded_resized_image(resized_size, resized_image.type(), cv::Scalar(0));
    cv::Rect roi((resized_size.width - new_size.width) / 2, (resized_size.height - new_size.height) / 2, new_size.width, new_size.height);
    resized_image.copyTo(padded_resized_image(roi));

    for (int i = 0; i < input_size[1]; i++) {
        for (int j = 0; j < input_size[2]; j++) {
            for (int k = 0; k < input_size[3]; k++) {
                input_tensor.data.f[i * input_size[2] * input_size[3] + j * input_size[3] + k] = static_cast<float>(padded_resized_image.at<cv::Vec3b>(i, j)[k]) / 255.0;
                // input_tensor.data.f[i * input_size[2] * input_size[3] + j * input_size[3] + k] = static_cast<float>(padded_resized_image.at<cv::Vec3b>(i, j)[k]);
            }
        }
    }

    ABSL_ASSERT(interpreter->Invoke() == kTfLiteOk);

    std::cout << "Output value:" << std::endl;
    for (int i = 0; i < output_size[1]; i++) {
        std::cout << "  - " << output_tensor.data.f[i] << std::endl;
    }

    return absl::OkStatus();
}


int main(int argc, char* argv[]) {
    if (!tflite::MMAPAllocation::IsSupported()) {
        LOG(ERROR) << "Memory mapping is not supported on this platform";
        return EXIT_FAILURE;
    }

    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    std::string model_path_str = absl::GetFlag(FLAGS_model_path);
    if (model_path_str.empty()) {
        LOG(ERROR) << "Please provide a model path, e.g., --model_path=/path/to/model.tflite";
        return EXIT_FAILURE;
    }
    std::filesystem::path model_path(model_path_str);
    if (!std::filesystem::exists(model_path)) {
        LOG(ERROR) << "Model file not found: " << model_path;
        return EXIT_FAILURE;
    }

    std::string image_path_str = absl::GetFlag(FLAGS_image_path);
    if (image_path_str.empty()) {
        LOG(ERROR) << "Please provide an image path, e.g., --image_path=/path/to/image.png";
        return EXIT_FAILURE;
    }
    std::filesystem::path image_path(image_path_str);
    if (!std::filesystem::exists(image_path)) {
        LOG(ERROR) << "Image file not found: " << image_path;
        return EXIT_FAILURE;
    }

    absl::Status status = Interpret(model_path, image_path);
    if (!status.ok()) {
        LOG(ERROR) << "Error: " << status.message();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

