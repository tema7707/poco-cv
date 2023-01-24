#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Util/ServerApplication.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <array>
#include <vector>

#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>

using namespace Poco;
using namespace Poco::Net;
using namespace Poco::Util;
using namespace cv;

constexpr int64_t numChannels = 3;
constexpr int64_t width = 224;
constexpr int64_t height = 224;
constexpr int64_t numClasses = 1000;
constexpr int64_t numInputElements = numChannels * height * width;


class ImageHandler: public Poco::Net::HTTPRequestHandler
{
public:
    void handleRequest(Poco::Net::HTTPServerRequest& request, Poco::Net::HTTPServerResponse& response)
    {
        if (request.getMethod() == Poco::Net::HTTPRequest::HTTP_POST)
        {
            try {
                // Read the image data from the request
                std::vector<uchar> buffer;
                std::copy(std::istreambuf_iterator<char>(request.stream()), std::istreambuf_iterator<char>(),
                          std::back_inserter(buffer));
                Mat image = imdecode(buffer, IMREAD_COLOR);
                // Verify that the image was decoded correctly
                if (image.empty()) {
                    std::cout << "Error decoding image" << std::endl;
                    return;
                }
                // Preprocess the image
                cv::resize(image, image, cv::Size(224, 224));
                cv::subtract(image, Scalar(0.485, 0.456, 0.406), image);
                cv::divide(image, Scalar(0.229, 0.224, 0.225), image);

                image = image.reshape(1, 1);
                std::vector<float> vec;
                image.convertTo(vec, CV_32FC1, 1. / 255);
                std::vector<float> array;
                for (size_t ch = 0; ch < 3; ++ch) {
                    for (size_t i = ch; i < vec.size(); i += 3) {
                        array.emplace_back(vec[i]);
                    }
                }

                // Convert image to std::vector<float>
                Ort::Env env;
                Ort::RunOptions runOptions;
                Ort::Session session(nullptr);

//                auto modelPath = "../mobilenetv2.onnx";
                auto modelPath = "assets/resnet50.onnx";
                const std::string labelFile = "assets/imagenet_classes.txt";
                //load labels
                std::vector<std::string> labels;
                std::ifstream file(labelFile);
                if (file) {
                    std::string s;
                    while (getline(file, s)) {
                        labels.emplace_back(s);
                    }
                    file.close();
                }
                if (labels.empty()) {
                    std::cout << "Failed to load labels: " << labelFile << std::endl;
                    return;
                }
                // create session
                Ort::SessionOptions session_options;
                session = Ort::Session(env, modelPath,session_options);
                // define shape
                const std::array<int64_t, 4> inputShape = { 1, numChannels, height, width };
                const std::array<int64_t, 2> outputShape = { 1, numClasses };
                // define array
                std::vector<float> input(numInputElements);
                std::vector<float> results(numClasses);

                // define Tensor
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                auto inputTensor = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), inputShape.data(), inputShape.size());
                auto outputTensor = Ort::Value::CreateTensor<float>(memory_info, results.data(), results.size(), outputShape.data(), outputShape.size());
                // copy image data to input array
                std::copy(array.begin(), array.end(), input.begin());
                // define names
                Ort::AllocatorWithDefaultOptions ort_alloc;
                Ort::AllocatedStringPtr inputName = session.GetInputNameAllocated(0, ort_alloc);
                Ort::AllocatedStringPtr outputName = session.GetOutputNameAllocated(0, ort_alloc);
                const std::array<const char*, 1> inputNames = { inputName.get()};
                const std::array<const char*, 1> outputNames = { outputName.get()};
                inputName.release();
                outputName.release();
                // run inference
                try {
                    session.Run(runOptions, inputNames.data(), &inputTensor, 1, outputNames.data(), &outputTensor, 1);
                }
                catch (Ort::Exception& e) {
                    std::cout << e.what() << std::endl;
                    return;
                }
                // sort results
                std::vector<std::pair<size_t, float>> indexValuePairs;
                for (size_t i = 0; i < results.size(); ++i) {
                    indexValuePairs.emplace_back(i, results[i]);
                }
                std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
                std::string answer = "";
                // show Top5
                for (size_t i = 0; i < 5; ++i) {
                    const auto& result = indexValuePairs[i];
                    answer += std::to_string(i + 1) + ": " + labels[result.first] + " " + std::to_string(result.second) + "\n";
                }

                // Set the response status and message
                response.setStatus(HTTPResponse::HTTP_OK);
                response.setContentType("text/plain");
                response.send() << answer;
            }
            catch (const std::exception& e)
            {
                std::cout << "Error: " << e.what() << std::endl;
                // Set the response status and message
                response.setStatus(HTTPResponse::HTTP_INTERNAL_SERVER_ERROR);
                response.setContentType("text/html");
                response.send() << "Error saving image";
            }
        }
        else
        {
            response.setStatus(Poco::Net::HTTPResponse::HTTP_BAD_REQUEST);
            response.setContentType("text/plain");
            std::ostream& ostr = response.send();
            ostr << "Only POST requests are accepted.";
        }
    }
};


class ImageHandlerFactory: public Poco::Net::HTTPRequestHandlerFactory
{
public:
    Poco::Net::HTTPRequestHandler* createRequestHandler(const Poco::Net::HTTPServerRequest& request)
    {
        return new ImageHandler;
    }
};

class WebServerApp: public ServerApplication
{
    void initialize(Application& self)
    {
        loadConfiguration();
        ServerApplication::initialize(self);
    }

    int main(const std::vector<std::string>&)
    {
        UInt16 port = static_cast<UInt16>(config().getUInt("port", 9090));
        HTTPServer srv(new ImageHandlerFactory, port);
        srv.start();
        logger().information("HTTP Server started on port %hu.", port);
        waitForTerminationRequest();
        logger().information("Stopping HTTP Server...");
        srv.stop();

        return Application::EXIT_OK;
    }
};

POCO_SERVER_MAIN(WebServerApp)
