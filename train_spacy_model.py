#!/usr/bin/env python3
"""
Train spaCy Model Script

This script trains a spaCy NER model using the training data generated
from the control descriptions.

Usage:
    python train_spacy_model.py train.spacy dev.spacy --output-dir ./models

Input:
    - Training data file (.spacy format)
    - Development data file (.spacy format)

Output:
    - Trained spaCy model
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
import shutil
import subprocess
import tempfile


def create_config(output_dir, base_model="en_core_web_md", gpu=False):
    """
    Create a spaCy training configuration.

    Args:
        output_dir (str): Directory to save the config
        base_model (str): Base model to use for training
        gpu (bool): Whether to use GPU for training

    Returns:
        str: Path to the config file
    """
    # Create config directory
    config_dir = Path(output_dir)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Create config file
    config_path = config_dir / "config.cfg"

    # Basic config template
    config = f"""
[paths]
train = null
dev = null
vectors = null
init_tok2vec = null

[system]
gpu_allocator = "{gpu_allocator}" 
seed = 0

[nlp]
lang = "en"
pipeline = ["tok2vec", "ner"]
batch_size = 1000
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {{"@tokenizers":"spacy.Tokenizer.v1"}}

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"
pretrained_vectors = null
normalize = false

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 96
attrs = ["ORTH", "SHAPE"]
rows = [5000, 2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 96
depth = 4
window_size = 1
maxout_pieces = 3

[components.ner]
factory = "ner"
incorrect_spans_key = null
moves = null
scorer = {{"@scorers":"spacy.ner_scorer.v1"}}
update_with_oracle_cut_size = 100

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = 96
upstream = "tok2vec"

[corpora]

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${train_path}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${dev_path}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = 0
gpu_allocator = "{gpu_allocator}"
dropout = 0.1
accumulate_gradient = 1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 0.00005

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[pretraining]

[initialize]
vectors = "{vectors}"
init_tok2vec = ${init_tok2vec}
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
    """

    # Set GPU allocator if using GPU
    gpu_allocator = "pytorch" if gpu else "null"

    # Set vectors and init_tok2vec based on base model
    vectors = f"{base_model}" if base_model else "null"
    init_tok2vec = "null"

    # Set train and dev paths in the config
    train_path = "${paths.train}"
    dev_path = "${paths.dev}"

    # Format the config with the variables
    config = config.format(
        gpu_allocator=gpu_allocator,
        train_path=train_path,
        dev_path=dev_path,
        vectors=vectors,
        init_tok2vec=init_tok2vec
    )

    # Write config to file
    with open(config_path, 'w') as f:
        f.write(config)

    return str(config_path)


def train_model(train_file, dev_file, output_dir, base_model="en_core_web_md", gpu=False, n_iter=20):
    """
    Train a spaCy NER model.

    Args:
        train_file (str): Path to training data file
        dev_file (str): Path to development data file
        output_dir (str): Directory to save the model
        base_model (str): Base model to use for training
        gpu (bool): Whether to use GPU for training
        n_iter (int): Number of training iterations

    Returns:
        bool: True if training was successful
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create config file
    config_path = create_config(output_dir, base_model, gpu)

    # Create a temporary directory for training
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the spaCy project command
        train_command = [
            "python", "-m", "spacy", "train",
            config_path,
            "--output", output_dir,
            "--paths.train", train_file,
            "--paths.dev", dev_file,
            "--training.max_steps", str(n_iter * 1000),
            "--training.eval_frequency", "100"
        ]

        if base_model:
            # Initialize with a base model
            train_command.extend(["--initialize.vectors", base_model])

        if gpu:
            # Use GPU for training
            train_command.extend(["--gpu-id", "0"])

        print(f"Running command: {' '.join(train_command)}")

        try:
            # Run the training command
            result = subprocess.run(train_command, check=True)

            if result.returncode == 0:
                print(f"Training successful. Model saved to {output_dir}")
                return True
            else:
                print(f"Training failed with return code {result.returncode}")
                return False

        except subprocess.CalledProcessError as e:
            print(f"Training failed with error: {e}")
            return False

        except Exception as e:
            print(f"Error during training: {e}")
            return False


def evaluate_model(model_dir, test_file=None):
    """
    Evaluate a trained spaCy model.

    Args:
        model_dir (str): Directory containing the trained model
        test_file (str, optional): Path to test data file

    Returns:
        dict: Evaluation results
    """
    # Load the trained model
    try:
        import spacy
        model_path = Path(model_dir) / "model-best"

        if not model_path.exists():
            print(f"Model not found at {model_path}")
            return None

        nlp = spacy.load(model_path)

        if test_file:
            # Evaluate on test data
            evaluate_command = [
                "python", "-m", "spacy", "evaluate",
                model_path,
                test_file
            ]

            print(f"Running evaluation: {' '.join(evaluate_command)}")

            try:
                result = subprocess.run(
                    evaluate_command,
                    check=True,
                    capture_output=True,
                    text=True
                )

                print(result.stdout)

                # Extract results from output
                results = {}

                for line in result.stdout.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        results[key.strip()] = value.strip()

                return results

            except subprocess.CalledProcessError as e:
                print(f"Evaluation failed with error: {e}")
                print(e.stdout)
                print(e.stderr)
                return None
        else:
            # Just return model info
            return {
                "model": model_path,
                "pipeline": nlp.pipe_names
            }

    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a spaCy NER model")
    parser.add_argument("train_file", help="Path to training data file (.spacy format)")
    parser.add_argument("dev_file", help="Path to development data file (.spacy format)")
    parser.add_argument("--output-dir", required=True, help="Directory to save the model")
    parser.add_argument("--base-model", default="en_core_web_md",
                        help="Base model to use for training (default: en_core_web_md)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--iterations", type=int, default=20, help="Number of training iterations (default: 20)")
    parser.add_argument("--test-file", help="Path to test data file for evaluation (.spacy format)")
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    # Check if required files exist
    for file_path in [args.train_file, args.dev_file]:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return 1

    if args.test_file and not os.path.exists(args.test_file):
        print(f"Error: Test file not found: {args.test_file}")
        return 1

    # Train the model
    success = train_model(
        args.train_file,
        args.dev_file,
        args.output_dir,
        args.base_model,
        args.gpu,
        args.iterations
    )

    if not success:
        print("Training failed")
        return 1

    # Evaluate the model
    if args.test_file:
        print("\nEvaluating model...")
        results = evaluate_model(args.output_dir, args.test_file)

        if results:
            print("\nEvaluation results:")
            for key, value in results.items():
                print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())