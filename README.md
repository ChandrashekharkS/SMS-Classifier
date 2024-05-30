# SMS Classifier

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![GitHub issues](https://img.shields.io/github/issues/yourusername/sms-classifier)
![GitHub stars](https://img.shields.io/github/stars/yourusername/sms-classifier)
![GitHub license](https://img.shields.io/github/license/yourusername/sms-classifier)

## Overview

The SMS Classifier is a machine learning project that identifies and classifies SMS messages as either spam or ham (non-spam). This project utilizes Python and various machine learning libraries to create a model capable of distinguishing between unwanted and genuine messages.

## Features

- **Data Preprocessing**: Clean and preprocess SMS data for optimal model performance.
- **Model Training**: Train a machine learning model using popular algorithms.
- **Evaluation Metrics**: Assess the model's performance using metrics like accuracy, precision, recall, and F1-score.
- **Prediction**: Classify new SMS messages as spam or ham.
- **Visualization**: Visualize data distribution and model performance using graphs.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/sms-classifier.git
    ```
2. Change to the project directory:
    ```bash
    cd sms-classifier
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Open and run the `SMS_Classifier.ipynb` notebook in Google Colab or any Jupyter environment.
2. Follow the instructions in the notebook to preprocess the data, train the model, and make predictions.

## Dataset

The dataset used for this project is a collection of SMS messages labeled as spam or ham. You can find and download the dataset [here](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

## Model

The project uses a combination of text preprocessing techniques and machine learning algorithms such as Naive Bayes, Support Vector Machine, and others to classify the messages. 

### Preprocessing Steps:

- Tokenization
- Removing stop words
- Lemmatization
- Vectorization using TF-IDF

### Algorithms:

- Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression

## Results

The model's performance is evaluated using the following metrics:

- **Accuracy**: The percentage of correctly classified messages.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all observations in the actual class.
- **F1-Score**: The weighted average of Precision and Recall.

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 98%     |
| Precision | 97%     |
| Recall    | 95%     |
| F1-Score  | 96%     |

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or issues, please contact:

- **Name**: Chandrashekhar K S
- **Email**: cg4618651@gmail.com
- **GitHub**: [ChandrasekharkS](https://github.com/ChandrasekharkS)
