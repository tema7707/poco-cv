#include "onnx_runner.h"

Runner::Runner() { }

Runner::Runner(const char* modelPath)
{
    // Create session
    Ort::Env env;
    Ort::SessionOptions session_options;
    session_ = Ort::Session(env, modelPath, session_options);
}

std::vector<float> Runner::inference(const Mat &image)
{
    std::vector<float> results(numClasses);
    if (image.empty()) {
        std::cout << "Error image is empty" << std::endl;
        return results;
    }

    std::vector<float> array = preprocess(image);
    std::vector<float> input(numInputElements);
    std::copy(array.begin(), array.end(), input.begin());

    // Define Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());

    // Define names
    Ort::AllocatorWithDefaultOptions ort_alloc;
    Ort::AllocatedStringPtr inputName = session_.GetInputNameAllocated(0, ort_alloc);
    Ort::AllocatedStringPtr outputName = session_.GetOutputNameAllocated(0, ort_alloc);
    const std::array<const char*, 1> inputNames = { inputName.get()};
    const std::array<const char*, 1> outputNames = { outputName.get()};
    inputName.release();
    outputName.release();

    // Run inference
    std::cout << "tuta";
    try {
        session_.Run(runOptions_, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
    }
    catch (Ort::Exception& e)
    {
        std::cout << "suda";
        std::cout << e.what() << std::endl;
        return results;
    }

    return results;
}

std::vector<float> Runner::preprocess(Mat image)
{
    // Preprocess the image
    cv::resize(image, image, cv::Size(width, height));
    cv::subtract(image, normalization_mean, image);
    cv::divide(image, normalization_variance, image);

    image = image.reshape(1, 1);
    std::vector<float> vec;
    image.convertTo(vec, CV_32FC1, 1. / 255);
    std::vector<float> array;
    for (size_t ch = 0; ch < 3; ++ch)
    {
        for (size_t i = ch; i < vec.size(); i += 3) {
            array.emplace_back(vec[i]);
        }
    }
    return array;
}
