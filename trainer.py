import argparse
from contextlib import contextmanager
import logging
import time

from smartsim.ml.torch import DynamicDataGenerator

import torch
import torch.nn as nn

import sklearn.preprocessing

@contextmanager
def timer(name):
    """ Measure and log the duration of code blocks.
    """
    start_time = time.perf_counter()
    yield
    logging.info(f"TIME ELAPSED {name} {time.perf_counter()-start_time}")

class MLP(nn.Module):
    """Simple artificial neural network
    """
    def __init__(self, num_layers, layer_width, input_size, output_size, activation_fn):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, layer_width))
        layers.append(activation_fn)

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(activation_fn)

        layers.append(nn.Linear(layer_width, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

def training_step(training_generator, X_scaler, optimizer, loss_function, epochs, model):
    """Does one round of a training step

    - Loop through the specified number of epochs.
    - Fetch batches of data from the training_generator.
    - Apply scaling to input features.
    - Compute predictions, loss, and backpropagates to update the model weights.
    - Log the cumulative loss at end of step for monitoring.

    :param training_generator: Data generator for the training data
    :param X_scaler: Pre-fit scaler to apply to new data
    :param optimizer: The optimizer to use for training
    :param loss_function: Loss function for training
    :param epochs: Number of epochs per loop
    :param model: The ANN to be trained
    :return: Updated ANN
    """
    with timer("Training step"):
        model.train()

        for epoch in range(epochs):
            summed_loss = 0
            nbatches = 0
            for X_batch, y_batch in training_generator:
                nbatches += 1
                X_batch = torch.tensor(X_scaler.transform(X_batch), dtype=torch.float32)
                y_batch = y_batch.unsqueeze(1)
                y_pred = model(X_batch)
                loss_value = loss_function(y_pred, y_batch)
                summed_loss += loss_value
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
        logging.info(f"Loss function: {summed_loss/nbatches}")

    return model

def save_model(model, example_input):
    model.eval()
    torch.save(model, "trained_model.pt")

def main(args):
    """Main loop for the trainer

    - Initialize logging and various components of the training setup (model, optimizer, loss function, data generator
    - Fit a scaler to the first batch of data to normalize input features
    - Enter an infinite loop that continually checks for new data, trains the model with available data,
      and saves it after each training cycle. If no new data is available, the training stops.
    """

    logging.basicConfig(level=logging.INFO)
    # Define the model and training-related objets
    model = MLP(
        args.num_layers,
        args.layer_width,
        args.nfeatures,
        1,
        nn.Sigmoid()
    )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logging.info("Initializing Data Generator")
    # Create the training dataset generator
    training_generator = DynamicDataGenerator(
        cluster=args.clustered_db,
        batch_size=args.batch_size,
        shuffle=True,
        data_info_or_list_name="training_data",
        verbose=True,
    )
    training_generator.init_samples()
    num_samples = training_generator.num_samples

    logging.info("Fitting scaler")
    # Create and initialize the scaler
    X_scaler = sklearn.preprocessing.MinMaxScaler()
    X_batch, y_batch = next(iter(training_generator))
    y_batch = y_batch.reshape(-1,1)
    X_scaler.fit(X_batch)

    # Enter training loop
    while True:

        logging.info("Beginning training loop")
        # Do a training loop
        model = training_step(
            training_generator,
            X_scaler,
            optimizer,
            loss_function,
            args.epochs,
            model
        )

        logging.info("Saving the model")
        # Save the latest model
        save_model(model, X_batch)

        # Avoid retraining unless new data is available
        waiting = True
        attempts = 0
        while waiting:
            training_generator.update_data()
            if training_generator.num_samples > num_samples:
                waiting = False
                logging.info("New data found")
                num_samples = training_generator.num_samples
                break
            else:
                time.sleep(0.5)
                attempts += 1
            if attempts == 20:
                break
        if attempts == 20:
            break

#A Set up and parse command-line arguments that configure the model training process,
#  such as whether the database is clustered, neural network parameters, and training configurations.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train data on the fly")
    parser.add_argument(
        "--clustered-db",
        action="store_true",
        help="True if the database is clustered"
    )
    parser.add_argument(
        "--num_layers",
        default=3,
        help="Number of layers in the neural network"
    )
    parser.add_argument(
        "--layer_width",
        default=20,
        help="Number of nodes within a layer"
    )
    parser.add_argument(
        "--nfeatures",
        default=3,
        help="Number of features for training"
    )
    parser.add_argument(
        "--batch-size",
        default=10,
        help="The batch size to train the model on"
    )
    parser.add_argument(
        "--learning-rate",
        default=1e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--epochs",
        default=100,
        help="Number of epochs"
    )
    args = parser.parse_args()
    main(args)