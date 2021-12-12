# Adhoc Custom Image Classification
This is a simple project to create an adhoc custom image classifier. It furthermore allows you to create a custom dataset with a few clicks only. The image classification itself is based on Tensorflow and the MobileNetv2 in particular.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. In the next 

### Prerequisites

The project requires `python3`, `pip3` and `virtualenv` to build and run. Furthermore, OpenCV needs to be installed on your system. For more information about how to install OpenCV on Mac or Linux visit <br />
[(Mac) https://docs.opencv.org/4.x/d0/db2/tutorial_macos_install.html](https://docs.opencv.org/4.x/d0/db2/tutorial_macos_install.html) <br />
[(Linux) https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html](https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html).


### Installing

Clone the project to your local machine.

```
git clone https://github.com/ollibeyer/adhoc-custom-image-classification.git
cd adhoc-custom-image-classification/
```

Create a virtual environment and activate it.

```
python3 -m venv tf-env
source tf-env/bin/activate
```

Install all Python requirements.

```
pip install -U pip
pip install -r requirements.txt
```

*Note: If you are using an Apple M1 based system, please use `requirements_m1.txt`*

## Create a custom dataset for training

In order to train your image classifier, we need to take some images of our objects in question. In the following, both the steps taking images using the webcam and name the categories are covered.

### Capturing images using your webcam

When running `python capture.py` a window will open showing your webcam feed. To capture a training image, you have to hold the object in question into the camera and press one of the numerical keys (0-9). Each key stands for a separate category. So if you have two objects you want to distinguish, for example, you could take pictures of them pressing the key `1` or `2` respectively. You can end the script by pressing `ESC` or `q`.

### Naming your categories

After capturing the images, a new folder `dataset` with the subfolders of the keys you have pressed will be generated automatically. You can now rename the subfolders (0-9) according to the representative category label.

## Train your model

In this step you can train your model by running `python training.py`. The script will automatically use the data stored in `dataset` and divide it 80/20 into training and validation data. The configuration file `config.json` can be modifed to adjust some training parameters. When the training is complete, a plot of the accuracy on the training and validation data throughout the training process is shown. The final model will be stored as `model.h5` and its label map is stored as `labels.txt`.

## Test your model

You can test your model by running `python inference_webcam.py`. The script will show your webcam feed again with additional information about the current fps as well as the found categories that have a confidence greater than `threshold` that is defined in `config.json`.

## Known Issues

/

## Authors

* [Oliver Beyer](https://github.com/ollibeyer)
