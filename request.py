from argparse import ArgumentParser
import requests


def main(file_path, url):
    # Open the image file
    with open(file_path, "rb") as image_file:
        image_data = image_file.read()
        # Send the image file to the server
        response = requests.post(url, data=image_data)

    # Check the status code of the response
    if response.status_code == 200:
        print(response.text)
    else:
        print("Error saving image")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--image_path', nargs='?', const=1, type=str, default="image.jpg")
    parser.add_argument('--url', nargs='?', const=1, type=str, default="http://localhost:9090")
    args = parser.parse_args()
    main(args.image_path, args.url)
