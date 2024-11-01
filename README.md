# Amazon_Sentiment_Analysis_And_NumberPlateRemoval_g4

## Car License Plate Identification and Blurring

This repository contains a model for detecting and blurring car license plates in images. It can be used to obscure license plates for privacy purposes. Follow the steps below to set up and use the model.

### Project Overview
The Car License Plate Identification and Blurring model identifies license plates in car images and applies a blurring effect to conceal sensitive information. This repository includes the necessary code and instructions for setting up and running the model.

### Getting Started

### 1. Clone the Repository
First, clone this repository to your local environment by running:

```bash
git clone https://github.com/realBagher/Amazon_Sentiment_Analysis_And_NumberPlateRemoval_g4.git
```

### 2. Download the Pre-trained Model
Download the pre-trained model file from [this link](https://drive.google.com/file/d/1rtCTLDdEkBxdfsvzc9TS31Z0HXiTTP6X/view?usp=drive_link) and save it in the `Model` folder in the cloned repository.

### 3. Run the Model
To execute the model on a set of images, run one of the following commands:

```bash
python blurring.py --input "/path/to/images" --output "/path/to/save/blurred/images" --skip-night true
```
This command will skip blurring images taken at night.

```bash
python blurring.py --input "/path/to/images" --output "/path/to/save/blurred/images" --skip-night false
```
This command will process all images, including those taken at night.

```bash
python blurring.py --input "/path/to/images" --output "/path/to/save/blurred/images"
```
If the `--skip-night` option is not specified, it defaults to `True`, meaning images taken at night will be skipped.


Replace `/path/to/images` with the path to your images folder and `/path/to/save/blurred/images` with the desired output path.

### Requirements
- Python 3.7+
- Keras
- OpenCV
- TensorFlow

### Contributing
Feel free to contribute to this project! Please fork the repository and create a pull request with your changes.
Here’s the README with the updated repository link:
