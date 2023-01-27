#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Util/ServerApplication.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <array>
#include <vector>

#include "onnx_runner.h"


using namespace Poco;
using namespace Poco::Net;
using namespace Poco::Util;
using namespace cv;


class ImageRequestHandler: public Poco::Net::HTTPRequestHandler
{
public:
    ImageRequestHandler(Runner* runner, std::vector<std::string>* labels)
    {
        runner_ = runner;
        labels_ = labels;
    }

    void handleRequest(Poco::Net::HTTPServerRequest& request, Poco::Net::HTTPServerResponse& response)
    {
        if (request.getMethod() != Poco::Net::HTTPRequest::HTTP_POST)
        {
            response.setStatus(Poco::Net::HTTPResponse::HTTP_BAD_REQUEST);
            response.setContentType("text/plain");
            response.send()  << "Only POST requests are accepted.";
            return;
        }
        try
        {
            // Read the image data from the request
            std::vector<uchar> buffer;
            std::copy(std::istreambuf_iterator<char>(request.stream()), std::istreambuf_iterator<char>(),
                      std::back_inserter(buffer));
            Mat image = imdecode(buffer, IMREAD_COLOR);
            std::vector<float> results = runner_->inference(image);

            std::string answer = interpret_results(results);
            response.setStatus(HTTPResponse::HTTP_OK);
            response.setContentType("text/plain");
            response.send() << answer;
        }
        catch (const std::exception& e)
        {
            std::cout << "Error: " << e.what() << std::endl;
            response.setStatus(HTTPResponse::HTTP_INTERNAL_SERVER_ERROR);
            response.setContentType("text/plain");
            response.send() << "Error saving image";
        }
    }

private:
    std::string interpret_results(std::vector<float> results, int n_top = 5)
    {
        std::string answer;
        std::vector<std::pair<size_t, float>> indexValuePairs;
        for (size_t i = 0; i < results.size(); ++i) {
            indexValuePairs.emplace_back(i, results[i]);
        }

        std::sort(indexValuePairs.begin(), indexValuePairs.end(), [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
        for (size_t i = 0; i < n_top; ++i)
        {
            const auto& result = indexValuePairs[i];
            answer += std::to_string(i + 1) + ": " + (*labels_)[result.first] + " " + std::to_string(result.second) + "\n";
        }
        return answer;
    }

    Runner* runner_;
    std::vector<std::string>* labels_;
};


class ImageRequestHandlerFactory: public Poco::Net::HTTPRequestHandlerFactory
{
public:
    ImageRequestHandlerFactory() {
        const std::string labelFile = "assets/imagenet_classes.txt";
        labels_ = generate_labels(labelFile);
        std::cout << "labels generated";
        auto modelPath = "assets/resnet50.onnx";
        runner_ = Runner(modelPath);
    }

    Poco::Net::HTTPRequestHandler* createRequestHandler(const Poco::Net::HTTPServerRequest& request)
    {
        return new ImageRequestHandler(&runner_, &labels_);
    }

private:
    static std::vector<std::string> generate_labels(const std::string& labelFile)
    {
        std::vector<std::string> generated_labels;
        std::ifstream file(labelFile);
        if (file)
        {
            std::string s;
            while (getline(file, s)) {
                generated_labels.emplace_back(s);
            }
            file.close();
        }
        if (generated_labels.empty())
        {
            std::cout << "Failed to load labels: " << labelFile << std::endl;
            return generated_labels;
        }
        return generated_labels;
    }

    Runner runner_;
    std::vector<std::string> labels_;
};

class WebServerApp: public ServerApplication
{
    int main(const std::vector<std::string>&)
    {
        UInt16 port = static_cast<UInt16>(config().getUInt("port", 9090));
        HTTPServer srv(new ImageRequestHandlerFactory, port);
        srv.start();
        logger().information("HTTP Server started on port %hu.", port);
        waitForTerminationRequest();
        logger().information("Stopping HTTP Server...");
        srv.stop();
        return Application::EXIT_OK;
    }
};

POCO_SERVER_MAIN(WebServerApp)
