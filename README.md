# Rubricnet

Paper in review. F. Author, S. Author, and T. Author, “Towards Explainable
and Interpretable Musical Difficulty Estimation: A parameter-efficient
approach”, in Proc. of the 25th Int. Society for Music Information Re-
trieval Conf., San Francisco, USA, 2024

## Disclaimer

>> Please note that this repository is currently a work in progress. The final code and its components are under active development and may undergo significant changes for usability. The final, stable version of the project will be uploaded after its formal acceptance for future use. We encourage users to check back regularly for updates and improvements.


## Abstract

Estimating music piece difficulty is important for organizing educational music collections. This process could be partially \us{automatized} to facilitate the educator's role. Nevertheless, the decisions performed by prevalent deep-learning models are hardly understandable, which may impair the acceptance of such a technology in music education curricula. Our work employs explainable descriptors for difficulty estimation in  symbolic music representations. Furthermore, through a novel parameter-efficient white-box model, we outperform previous efforts while delivering interpretable results. These comprehensible outcomes emulate the functionality of a rubric, a tool widely used in music education.
Our approach, evaluated in piano repertoire categorized in 9 classes, achieved  
41.5% accuracy independently, with a mean squared error (MSE) of 1.52, showing precise difficulty estimation. 
Through our baseline, we
illustrate how building on top of past research can  offer alternatives for music difficulty assessment which are explainable and interpretable. With this, we aim to promote a more effective communication between the Music Information Retrieval (MIR) community and the music education one.

## Installation
### Prerequisites

- Python 3.6 or newer

- Pip



### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/anonymous/anonymous.git
   ```
   
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```



## Usage
### Directory Structure
- **`checkpoints/`**: Contains model checkpoints.
- **`features/`**: Stores different feature sets.
  - **`rubricnet/`**:
    - **`optuna_bayesian_optimization.py`**: Code used for training.
    - **`interpretability.py`**: interpretability feedback and results on CIPI. 
    - **`rubricnet.py`**: Main components of the RubricNet architecture and a class abstraction to use RubricNet as an sklearn model.