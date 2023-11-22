# Vietnamese Handwritten OCR Challenge

## Description

### Problem Statement

Optical Character Recognition (OCR) is a classic problem in the field of computer vision. Despite years of research, OCR for handwritten text, especially in Vietnamese, still presents significant challenges.

In this challenge, participating teams are tasked with building a lightweight OCR model to solve the problem of recognizing handwritten Vietnamese addresses.

### Input & Output

**Input:** 
- Raw images containing a line of handwritten Vietnamese address.

**Output:** 
- Extracted text from the input image.

### Examples

**Input Image:**

![Input Image](link_to_input_image)

**Label:** Thôn 1 Ea Ktur Cư Kuin Đắk Lắk

### Evaluation

#### Stage 1

In the first stage, team solutions are evaluated based on the edit distance metric (Levenshtein distance). Specifically, for each test case, the team's score is calculated as follows:

Assuming the team's result is "output" and the correct label is "label":

- If the result is perfectly accurate (output == label, edit distance is 0), the team receives a score of 1.
- Otherwise, the score for the test case is calculated using the formula: score = max(0, 1 - 1.5^d/n)

where d is the edit distance between the output and label strings, and n is the length of the label string.

The team's overall score is the average score across all test cases. Teams with higher scores are ranked higher.

In case of ties, the inference model's execution time is considered. Details about resource limitations and inference time are specified in the Limitation section.

The first stage of the competition consists of two phases:

1. **Public Test:** Teams submit only the CSV result file as described. Public test scores are for reference only.
2. **Private Test:** Teams submit the source code, and the organizers will directly run the inference on the provided server. Final rankings are determined based on private test results.

For more information on edit distance, refer to [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance).

#### Stage 2

Coming soon

### Limitation

- **Model Size:** ≤ 50MB
- **Inference Time:** 2 seconds
- **Inference Environment:** No internet connection
- **Machine Configuration:** 
- CPU: Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz
- RAM: 64GB
- **Data:** Teams are not allowed to use datasets other than those provided in the competition, except for self-generated datasets.
- **Pre-trained Model:** Only publicly available pre-trained models trained on the ImageNet dataset are allowed. Using pre-trained models specifically trained for OCR tasks (whether for printed or handwritten text) is not permitted.

## Getting Started

[Instructions for setting up and running the OCR model will be provided in Stage 2.]


