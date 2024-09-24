import argparse
import logging
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import scipy

from smartredis import Client, Dataset
from smartsim.ml import TrainingDataUploader

class BatchList(list):
    """A custom list-like class for popping off batches of items"""
    def pop_batch(self, batch_length: int):
        """Pop off exactly n elements from a list

        :param batch_length: The size of the batch to create
        :return: A list with the requested
        """

        if len(self) >= batch_length:
            return [self.pop() for _ in range(batch_length)]
        return []

    def retrieve_batches(self, batch_length):
        """"""
        batches = []
        while new_batch := self.pop_batch(batch_length):
            batches.append(new_batch)
        return batches


@contextmanager
def timer(name):
    """ Measure and log the duration of code blocks.
    """

    start_time = time.perf_counter()
    yield
    logging.info(f"TIME ELAPSED: {name} {time.perf_counter()-start_time}s")

def convert_dataset_to_pandas(
    dataset,
    fields=[
        "avg_af",
        "avg_relV",
        "H_index",
        "avg_dPdy"
    ]
):
    """Convert a SmartSim dataset to a pandas dataframe
    """
    df = pd.DataFrame(
        {field:dataset.get_tensor(field) for field in fields}
    )
    return df

def downsample_dataset(fraction_samples, df, nbins=50):
    """Subsample the data based on the inverse PDF

    :param fraction_samples: Fraction to reduce the original data by
    :param df: Dataframe that holds th original dataset
    :return: Subsampled dataframe
    """

    pdf, bins, bin_number = scipy.stats.binned_statistic_dd(df.values, None, 'count', bins=nbins, expand_binnumbers=True)
    pdf = pdf/pdf.sum()
    pdf_transform = np.reciprocal(pdf, where=pdf>0.)

    bin_number = bin_number - 1
    weights = [pdf_transform[tuple(indices)] for indices in bin_number.transpose()]
    nsamples = round(fraction_samples * len(df))
    return df.sample(nsamples, weights=weights, replace=False)

def update_available_datasets(client, dataset_list, available_data):
    """Update the available datsaets for the given ensemble member
    """

    if client.get_list_length(dataset_list) > 0:
        temporary_list = f"temporary_list"
        # Rename this to avoid race condition with MFIX-Exa updating the list
        client.rename_list(dataset_list, temporary_list)
        new_datasets = client.get_datasets_from_list(temporary_list)
        available_data += new_datasets
        # Delete datasets from database to free up memory
        map(client.delete_dataset, new_datasets)
        logging.info(f"Simulation dataset updated with {len(new_datasets)} datasets")

def main(args):
    """Downsample the data

    - Set up basic logging.
    - Initializes a SmartRedis client for database interaction
    - Set up a data uploader (to manage the uploading of downsampled data for subsequent processing or training)
    - Infinite Loop: Continuously checks for new datasets, processes them if available, and downsamples them
        includes robust handling of waiting conditions and shutdown criteria based on data availability and other conditions.

    :param args: ArgParsed arguments
    """

    logging.basicConfig(level=logging.INFO)

    logging.info("Initializing downsampler with following configuration")
    logging.info(args)

    logging.info("Initializing SmartRedis client")
    client = Client()

    uploader = TrainingDataUploader(
        cluster=args.clustered_db
    )
    uploader.publish_info()

    simulation_data = BatchList()

    # Create objects for every ensemble member
    logging.info("Beginning downsample loop")
    wait_attempts = 0

    while True:

        update_available_datasets(client, args.dataset_list, simulation_data)
        batches = simulation_data.retrieve_batches(args.num_timesteps)

        # Downsample any available subsets
        action_taken = False
        for batch_idx, batch in enumerate(batches):
            logging.info(
                f"Downsampling batch {batch_idx+1}/{len(batches)}"
            )
            df = pd.concat([convert_dataset_to_pandas(dataset) for dataset in batch])
            df.dropna(inplace=True)
            print(df.describe())
            downsampled_df = downsample_dataset(args.fraction_samples, df)
            uploader.put_batch(
                downsampled_df[args.features].to_numpy(),
                downsampled_df[args.target].to_numpy()
            )
            action_taken = True

        if action_taken:
            wait_attempts = 0
        else:
            wait_attempts += 1
            if wait_attempts == args.num_wait_attempts:
                logging.info("Reached the maximum number of wait attempts. Exiting")
                sys.exit()
            logging.info("No action taken...sleeping")
            time.sleep(args.wait_interval)

#A Argument Parsing and Script Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample data from MFIX-exa simulation")
    parser.add_argument(
        "--dataset-list",
        type=str,
        help="Name of the dataset list from the simulation",
        default="simulation_data"
    )
    parser.add_argument(
        "--fraction-samples",
        type=float,
        help="Fraction of data to downsample to",
        default=0.05
    )
    parser.add_argument(
        "--clustered-db",
        action="store_true",
        help="True if the database is clustered"
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        help="Number of timesteps to wait for before downsampling",
        default=8
    )
    parser.add_argument(
        "--num-wait-attempts",
        type=int,
        help="Number of attempts to find the dataset and to grow to the selected size",
        default=120
    )
    parser.add_argument(
        "--wait-interval",
        type=float,
        help="Number of attempts to find the dataset and to grow to the selected size",
        default=0.5
    )
    parser.add_argument(
        "--features",
        nargs="*",
        help="List of features to extract for training",
        default=["avg_af", "avg_relV", "avg_dPdy"]
    )
    parser.add_argument(
        "--target",
        default="H_index",
        help="Name of the field to predict",
    )

    args = parser.parse_args()
    args.num_wait_attempts = int(args.num_wait_attempts)
    main(args)