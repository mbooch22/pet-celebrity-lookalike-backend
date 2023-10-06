# Pet Celebrity Lookalike Backend

This backend server for the Pet Celebrity Lookalike application is responsible for handling image uploads, feature extraction, celebrity comparison, and model training.

## Files and Their Functions

### `celebrity_feature_extraction.py`

This script is used to extract features from images of celebrities in the dataset. The extracted features are stored in the `celebrity_features.json` file.

### `compare_features.py`

This script compares the features of user-uploaded pet images with the features of celebrities. It calculates the similarity scores and returns the top matches.

### `celebrity_features.json`

This JSON file contains the features extracted from images of celebrities. Each entry in the file represents a celebrity with their corresponding feature vector.

### `features.json`

This JSON file contains features extracted from pet images used for training the model. It is utilized in the training process.

### `feature_extraction.py`

This script is used for feature extraction from user-uploaded pet images. It utilizes a pre-trained ResNet18 model and saves the features in the `user_features.json` file.

### `model_train_extraction.py`

This script is used to train the model based on features extracted from pet images. It saves the trained model as `resnet18.pth`.

### `resnet18.pth`

This file contains the weights of the pre-trained ResNet18 model. It is loaded during feature extraction.

### `train_classifier.py`

This script trains a classifier on the features extracted from pet images. It uses the trained model for celebrity comparison in the application.

### `user_features.json`

This JSON file contains the features extracted from user-uploaded pet images. It is used in the comparison process.

### `server.js`

This is the main server file using Node.js and Express.js. It handles incoming requests from the frontend, manages image uploads, and communicates with the Python scripts for feature extraction and celebrity comparison.

### `dataset`

dataset not included. 

## Technologies Used

- **Node.js:** JavaScript runtime for server-side development.
- **Express.js:** Web application framework for Node.js.
- **Python:** Used for machine learning tasks and feature extraction.
- **PyTorch:** Machine learning library used for the ResNet18 model.

## Usage

1. Start the server by running `node server.js`.

2. The server listens for image upload requests from the frontend.

3. Feature extraction and celebrity comparison are performed using Python scripts.

4. The results are sent back to the frontend for display.

## Contributing

If you'd like to contribute to this project, feel free to submit pull requests or open issues. We welcome improvements to the backend logic, scripts, or server functionality.

## License

See the [LICENSE](https://choosealicense.com/no-permission/) file for details.
