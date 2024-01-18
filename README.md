
# Detection Transformer (Detr) Implementation

![Static Badge](https://img.shields.io/badge/Detection-Transforemr-%23F9E79F)
![Static Badge](https://img.shields.io/badge/Pytorch-%23D5F5E3)


This repository contains my custom implementation of the Detection Transformer (Detr), a state-of-the-art object detection model based on transformer architecture.

## Overview

Detr eliminates the need for traditional region proposal networks (RPNs) and anchor boxes, treating object detection as a set prediction problem. The transformer-based architecture allows for capturing global context and dependencies among different parts of the image simultaneously.

## Features

- **Transformer Architecture:** Leverages the power of transformers for capturing contextual information in object detection.
- **Set Prediction:** Directly predicts class labels and bounding boxes for all objects in the image.
- **Dynamic Attention:** Handles a variable number of objects without predefined anchor boxes, making it flexible across various scales and aspect ratios.
