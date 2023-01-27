#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

const int64_t numChannels = 3;
const int64_t width = 224;
const int64_t height = 224;
const int64_t numClasses = 1000;
const int64_t numInputElements = numChannels * height * width;

const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
const std::array<int64_t, 2> outputShape = { 1, numClasses };

const Scalar normalization_mean = Scalar(0.485, 0.456, 0.406);
const Scalar normalization_variance = Scalar(0.229, 0.224, 0.225);

class Runner {
public:
    explicit Runner();
    explicit Runner(const char* modelPath);

    std::vector<float> inference(const Mat& image);

private:
    static std::vector<float> preprocess(Mat image);

    Ort::RunOptions runOptions_;
    Ort::Session session_ = Ort::Session(nullptr);
};