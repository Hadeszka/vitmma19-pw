## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Horváth Dóra
- **Aiming for +1 Mark**: No

### Solution Description

Problem description.
The goal of this project is to automatically predict the readability/quality rating (on a 1–5 scale) of Hungarian legal text paragraphs. The task is formulated as a multi-class text classification problem, where each paragraph is assigned to one of five discrete classes. A key challenge is the strong class imbalance and the high level of lexical and syntactic similarity between legal texts, which makes the classification problem non-trivial.

Model architecture.
Several lightweight neural architectures were implemented and compared, including a simple MLP baseline, a CNN-based text classifier, and a more expressive BiLSTM model with an attention mechanism. All models rely on a trainable word embedding layer, followed by task-specific feature extraction and a final classification head. The BiLSTM with attention was selected for the final experiments, as it can better model long-range dependencies and complex sentence structures that are typical in legal language, while still keeping the model size manageable.

Training methodology.
The full pipeline includes automated data downloading, preprocessing, deduplication, and stratified splitting into train, validation, and test sets, with the test set constructed from consensus-annotated samples. A rule-based baseline is evaluated on the validation set for comparison. Neural models are trained using cross-entropy loss, optionally with class weighting to address class imbalance. Training is performed exclusively on CPU, as no GPU was available on the development machine. Despite this constraint, training remains feasible due to the relatively small model sizes and dataset. Training progress is logged epoch-by-epoch, including loss and accuracy on both training and validation sets, and early stopping is applied based on validation loss.

Results.
The neural models consistently outperform the rule-based baseline, demonstrating the benefit of learned representations over handcrafted rules. Although overall accuracy is limited by class imbalance and annotation noise, the models achieve improved macro-level performance and more informative confusion matrices. The final system supports reproducible experiments, evaluation on a held-out test set, and inference on unseen texts, all runnable in a CPU-only Docker environment.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.


#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t my-project .
```

#### Run

To run the solution, use the following command.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run --rm  -v ${PWD}\data:/app/data -v ${PWD}\models:/app/models my-project > run.log 2>&1
```
*   The > log/run.log 2>&1 part ensures that all output (standard output and errors) is saved to log/run.log.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions. If data is not given it uses the first n rows of the test data.
        Example: 
        ```bash
        docker run --rm -v ${PWD}\data:/app/data -v ${PWD}\models:/app/models my-project python src/04-inference.py --text "A szolgáltatás használatához internetkapcsolat szükséges."
        ```
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-check-duplications.ipynb`: Notebook for checking data duplication.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `run.sh`: Runs the full pipeline (preprocessing, training, evaluation).

