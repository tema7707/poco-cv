# POCO web server for image classification

This is a web server written in C++ using the POCO library and the ONNX runtime for image classification. The server can be run using Docker and tested using the included request.py script.

## Requirements
- Docker
- ONNX runtime
- POCO library
- OpenCV

## Usage
1. Build the Docker image by running `docker build -t poco_cv` . in the project directory.
2. Run the container using `docker run -p 9090:9090 -it poco_cv`.
3. You could test the server using the included **request.py** script.

## Model
This POCO web server uses a pre-trained ResNet50 model that was trained on the ImageNet dataset. All images are preprocessed by being resized to a size of 224px, normalized with a mean of [0.485, 0.456, 0.406], and a standard deviation of [0.229, 0.224, 0.225].

## API
The server provides a single endpoint for classifying images. The endpoint accepts a POST request with the image file attached as form data. The server will return a text with the top predictions of the model.


## License
This project is licensed under the [Apache 2.0 license](https://github.com/tema7707/poco-cv/blob/main/LICENSE).
