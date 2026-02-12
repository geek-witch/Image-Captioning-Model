# Image Captioning Model (Flickr30k: ResNet50 + LSTM)

This repository contains an end-to-end deep learning implementation of an image captioning system that generates natural language descriptions for images. The model combines a pre-trained ResNet50 convolutional neural network for visual feature extraction with an LSTM-based decoder for caption generation, trained on the Flickr30k dataset.

## Overview

Image captioning is a multimodal task that integrates computer vision and natural language processing. This project implements a complete pipeline that includes:

- Image preprocessing and feature extraction
- Caption cleaning and vocabulary construction
- Encoder–decoder neural network architecture
- Training and validation workflow
- Caption generation during inference
- BLEU score evaluation

The system learns to map visual features to coherent textual descriptions.

## Model Architecture

The model follows an encoder–decoder design:

- **Encoder:** A pre-trained ResNet50 network extracts high-level image features using transfer learning.
- **Decoder:** An LSTM-based sequence model generates captions word by word using learned embeddings.

Architecture flow:

```
Image -> ResNet50 Encoder -> Feature Vector -> LSTM Decoder -> Caption
```

## Technologies Used

- Python
- PyTorch and Torchvision
- NLTK
- NumPy and Pandas
- Matplotlib
- Jupyter Notebook

## Dataset

The model is trained on the Flickr30k dataset, which contains approximately 30,000 images, each paired with multiple human-written captions. The dataset enables the model to learn associations between visual content and natural language.

## Repository Structure

```
.
├── Image Captioning App.ipynb   # Main notebook containing the full pipeline
└── README.md
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Image-Captioning-Model.git
cd Image-Captioning-Model
```

2. Install required dependencies:

```bash
pip install torch torchvision nltk numpy pandas matplotlib
```

3. Download the Flickr30k dataset and configure the dataset paths in the notebook.

## Usage

Open and run the Jupyter notebook:

```
Image Captioning App.ipynb
```

The notebook guides you through feature extraction, preprocessing, training, and caption generation.

## Results

Model performance is evaluated using BLEU score metrics and qualitative comparison between predicted and ground-truth captions. The model is able to generate meaningful descriptions of image content.

## Future Improvements

Possible extensions include:

- Incorporating attention mechanisms
- Using transformer-based architectures
- Applying beam search for caption generation
- Training on larger datasets

## License

This project is released under the MIT License.

## Acknowledgments

- Flickr30k dataset contributors
- PyTorch community
- Research on CNN–RNN image captioning systems
