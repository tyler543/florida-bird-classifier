# Florida Bird Classifier
### Real-time bird species identification using transfer learning on a Raspberry Pi 5

A real-time bird species classifier trained on 40 Florida bird species using 
transfer learning on MobileNetV4. Deployed on a Raspberry Pi 5 with the 
Pi Camera Module 3 for live inference.

## Results
- **Validation Accuracy:** 88.68%
- **Species:** 40 Florida bird species
- **Model:** MobileNetV4 (mobilenet_v4_s_il-common)
- **Training:** Transfer learning with 80/20 train/validation split

## Files
- `train_validate.py` — trains and validates the model with an 80/20 split
- `train.py` — simplified training script without validation
- `model_with_camera.py` — runs live inference using a webcam
- `trained_model.py` — runs inference on a single static image
- `test_birder.py` — tests the raw pre-trained Birder model on a static image
- `config.py` — central configuration file, edit paths and settings here
- `requirements.txt` — project dependencies

## Setup

### Install dependencies
pip install -r requirements.txt

### Configure paths
Edit `config.py` to set your data directory, model path, and other settings.

DATA_DIR will be the folder your model trains from

MODEL_PATH will save the trained model with the name you can set
by default it will be demo_model.pth (make sure its a pth file)

TEST_IMAGE is the image it will test one for static classification
The image is a Mallard

### Train the model
python train_validate.py

### Run live inference
python model_with_camera.py

## Requirements
- Python 3.10+
- PyTorch
- OpenCV
- Birder
- Raspberry Pi 5 with Pi Camera Module 3 (for deployment)

## Credits
This project uses the [Birder](https://github.com/birder-project/birder) 
library for pre-trained models and inference utilities, licensed under 
the Apache 2.0 License.

## License
This project is licensed under the Apache 2.0 License — see the LICENSE 
file for details.