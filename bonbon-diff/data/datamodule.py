# TODO: reorder imports to go from most standard library, third party library, module library
from typing import Union, Optional, Iterator, Literal
from pathlib import Path
import time

import lightning as L
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from mochi.tasks import get_task, load_task_dataset, download_from_s3

from bonbontoken.data import BonbonTokenDataset, collate_bonbontoken, BonbonMLMDataset, collate_bonbonmlm
from bonbontoken.tokenizer import MODALITY_SETTINGS
from bonbontoken.utils import (check_selfies_format, load_task_dataset_with_length_filtering,
# TODO: add these to __all__ or refactor into a globals
                               RELEASE_BUCKET, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, SOURCE_COLUMN, SOURCE_PREFIX)


class BonbonTokenDataModule(L.LightningDataModule):
    """Lightning DataModule for handling modality pair datasets.

    This DataModule manages the loading, processing, and batching of modality pair
    data for training, validation, and testing. It supports multiple data sources:

    1. Loading from Mochi tasks (standardized datasets from S3)
    2. Loading from local tab-separated files
    3. Splitting a single dataset into train/val/test partitions

    The module handles sequence length filtering, positive pair filtering, and
    normalization of labels. It creates appropriate PyTorch DataLoaders with
    custom collation functions for efficient batching.

    Attributes:
        pairs_file: Path to the main data file containing modality pairs
        modality_1_column: Name of the column containing first modality identifiers
        modality_2_column: Name of the column containing second modality identifiers
        label_column: Name of the column containing interaction labels
        source_column: Name of the column containing data source information
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        max_modality_1_seq_length: Maximum allowed length for first modality sequences
        max_modality_2_seq_length: Maximum allowed length for second modality sequences
        train_dataset: Dataset object for training data
        val_dataset: Dataset object for validation data
        test_dataset: Dataset object for test data
    """
    def __init__(
        self,
        modality_1_column: Literal["Protein", "Protein_1", "Protein_2", "Molecule"],
        modality_2_column: Literal["Protein", "Protein_1", "Protein_2", "Molecule"],
        modality_1_seq_column: Literal["Sequence", "Sequence_1", "Sequence_2", "SELFIES", "SMILES"],
        modality_2_seq_column:  Literal["Sequence", "Sequence_1", "Sequence_2", "SELFIES", "SMILES"],
        max_modality_1_seq_length: int,
        max_modality_2_seq_length: int,
        modality_1_type: Literal['protein', 'molecule', 'protbert', 'esm2', 'selfiested', 'chemberta'],
        modality_2_type: Literal['protein', 'molecule', 'protbert', 'esm2', 'selfiested', 'chemberta'],
        label_column: str = "Label",
        pairs_file: Union[str, Path] = None,
        source_column: str = SOURCE_COLUMN,
        compute_source_metrics: bool = True,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),  # TODO redundant
        val_file: Union[str, Path] = None,
        test_file: Union[str, Path] = None,
        persistent_workers: bool = False,
        normalize_label: bool = False,
        pos_filtering: bool = False,
        # Backward compatibility for precalculated embeddings / pre-tokenized sequences
        modality_1_emb_path: Optional[Union[str, Path]] = None,
        modality_2_emb_path: Optional[Union[str, Path]] = None,
        mochi_task: str = None,
        mochi_task_type: Literal['task','evaluation_task'] = 'task',
        mochi_task_version: str = 'latest',
        nvme_folder: Union[str, Path] = '/opt/dlami/nvme/',
    ):
        """Initialize the BonbonTokenDataModule.

        Args:
            modality_1_column: Name of the first modality column in the dataframe
            modality_2_column: Name of the second modality column in the dataframe
            modality_1_seq_column: Name of the column containing sequences for the first modality
            modality_2_seq_column: Name of the column containing sequences for the second modality
            max_modality_1_seq_length: Maximum allowed length for first modality sequences
            max_modality_2_seq_length: Maximum allowed length for second modality sequences
            modality_1_type: Type of the first modality (protein, molecule, etc.)
            modality_2_type: Type of the second modality (protein, molecule, etc.)
            label_column: Name of the column containing interaction labels
            pairs_file: Path to the main data file containing modality pairs
            source_column: Name of the column containing data source information
            compute_source_metrics: Whether to compute metrics per data source
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
            train_val_test_split: Ratios for splitting data into train/val/test sets
            val_file: Path to validation data file (if separate from pairs_file)
            test_file: Path to test data file (if separate from pairs_file)
            persistent_workers: Whether to keep worker processes alive between epochs
            normalize_label: Whether to normalize labels across the dataset
            pos_filtering: Whether to filter out negative samples
            modality_1_emb_path: Path to pre-calculated embeddings for first modality
            modality_2_emb_path: Path to pre-calculated embeddings for second modality
            mochi_task: Name of the Mochi task to load data from
            mochi_task_type: Type of Mochi task ('task' or 'evaluation_task')
            mochi_task_version: Version of the Mochi task to use
            nvme_folder: Path to NVME folder for temporary storage
        """

        super().__init__()
        self.pairs_file = pairs_file
        self.modality_1_column = modality_1_column
        self.modality_2_column = modality_2_column
        self.label_column = label_column
        self.modality_1_seq_column = modality_1_seq_column
        self.modality_2_seq_column = modality_2_seq_column
        self.seq_smile_columns = {modality_1_seq_column, modality_2_seq_column}
        self.modality_1_type = modality_1_type
        self.modality_2_type = modality_2_type
        self.max_modality_1_seq_length = max_modality_1_seq_length
        self.max_modality_2_seq_length = max_modality_2_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.val_file = val_file
        self.test_file = test_file
        self.pos_filtering = pos_filtering
        self.persistent_workers = persistent_workers
        self.normalize_label = normalize_label
        self.modality_1_emb_path = modality_1_emb_path
        self.modality_2_emb_path = modality_2_emb_path

        self.mochi_task = mochi_task
        self.mochi_task_type = mochi_task_type
        self.mochi_task_version = mochi_task_version
        self.nvme_folder = nvme_folder

        # apply drop last not as a parameter while we are testing the idea
        self.drop_last = True

        # custom collate function
        self.collate_fn = collate_bonbontoken

        self.compute_source_metrics = compute_source_metrics
        self.source_column = source_column
        self.unique_sources = []  # Will be populated during source processing

        # Initialize source DataFrames for each split
        # self.train_sources_df = None
        # self.val_sources_df = None
        # self.test_sources_df = None

        self.save_hyperparameters()
        # initialize dataframes
        self.pairs_dataframe = None  # dataframe of main training data
        # initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # self.from_disk = False
        #
        # if self.modality_1_emb_path is not None or self.modality_2_emb_path is not None:
        #     if self.mochi_task is not None:
        #         raise ValueError("mochi_task is provided but modality_1_emb_path or modality_2_emb_path were also provided - "
        #                          "please provide only one of them.")
        #     else:
        #         self.from_disk = True
        #         print("Loading pre-calculated embeddings from disk. ")

    def process_sources(self, df: pd.DataFrame, split: str = 'train') -> Optional[list[dict]]:
        """Process source column into list of dicts format.

        This method checks if a source column exists in the DataFrame and converts
        it to a memory-efficient list of dicts format. Each dict represents one
        sample's source information in one-hot encoded format.

        Args:
            df: DataFrame potentially containing source column
            split: Dataset split ('train', 'val', or 'test'). Sources are only
                processed for val and test splits when compute_source_metrics is True.

        Returns:
            List of dicts where each dict contains source information for one sample,
            or None if source processing is not applicable.
        """
        # Skip source processing if not needed
        if not self.compute_source_metrics:
            return None

        # Only process sources for val and test splits
        if split == 'train':
            return None

        # Check if source column exists
        if self.source_column is None or self.source_column not in df.columns:
            print(f"Source column '{self.source_column}' not found in {split} data. "
                  f"Source-based metrics will not be computed.")
            return None

        print(f"Processing sources for {split} split from column '{self.source_column}'")
        df[self.source_column] = df[self.source_column].fillna('[]')  # Fill NaNs with empty list
        # Get one-hot encoding
        # parse the source column into a list of sources
        parsed_source_column = df[self.source_column].str.findall(
            r"(?:'|\")?([^,\s\[\]]+)(?:'|\")?"
        )

        if any(parsed_source_column.apply(type) != list):
            raise ValueError(f"Source column '{self.source_column}' is not in the expected format.")

        source_dummies = parsed_source_column.str.join('|').str.get_dummies()
        source_dummies.columns = [SOURCE_PREFIX + column for column in source_dummies.columns]

        # Update unique sources list
        if split in ['val', 'test']:
            new_sources = source_dummies.columns.tolist()
            # Merge with existing unique sources
            self.unique_sources = sorted(list(set(self.unique_sources + new_sources)))
            print(f"Found {len(new_sources)} unique sources in {split} data")

        # Convert to list of dicts for memory efficiency
        # orient='records' creates a list where each element is a dict for one row
        sources_list = source_dummies.to_dict(orient='records')

        return sources_list

    def filter_positives(self, df: pd.DataFrame, label_column: str) -> pd.DataFrame:
        """Filter out positive pairs from the dataset if pos_filtering is enabled.

        This method is called during initialization to ensure filtering is consistent
        across all processes.

        Args:
            df: DataFrame containing the data
            label_column: Name of the column containing labels

        Returns:
            DataFrame with positive pairs filtered out if pos_filtering is True,
            otherwise returns the original DataFrame
        """
        # Filter positive pairs during initialization to ensure it's available across all processes
        if self.pos_filtering:
            original_length = len(df)
            df = df[df[label_column] > 0]
            print(f"Filtered {original_length - len(df)} positive pairs out of {original_length} pairs")
        return df

    def normalize(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Normalize the label column in all datasets using training set statistics.

        Calculates mean and standard deviation from the training set and applies
        the same normalization to all datasets to ensure consistent scaling.

        Args:
            train_df: Training dataset DataFrame
            val_df: Validation dataset DataFrame
            test_df: Test dataset DataFrame

        Returns:
            Tuple of normalized (train_df, val_df, test_df) DataFrames
        """
        label_mean = train_df[self.label_column].mean()
        label_std = train_df[self.label_column].std()
        train_df[self.label_column] = (train_df[self.label_column] - label_mean) / label_std
        val_df[self.label_column] = (val_df[self.label_column] - label_mean) / label_std
        test_df[self.label_column] = (test_df[self.label_column] - label_mean) / label_std
        return train_df, val_df, test_df

    def reduce_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the DataFrame to include only sequences within the maximum length limits.

        Removes rows where either modality 1 or modality 2 sequences exceed the maximum
        allowed lengths specified in the configuration.

        Args:
            df: DataFrame containing modality 1 and modality 2 sequences

        Returns:
            Filtered DataFrame containing only sequences within length limits

        Raises:
            ValueError: If required sequence columns are missing from the DataFrame
        """
        if self.seq_smile_columns.issubset(df.columns):
            condition = (df[self.modality_1_seq_column].str.len() < self.max_modality_1_seq_length) & (
                    df[self.modality_2_seq_column].str.len() < self.max_modality_2_seq_length)
            return df[condition]
        else:
            raise ValueError(f"Missing required columns: {self.seq_smile_columns}. Dataframe columns are: {df.columns}")

    def verify_data(self) -> None:
        """Verify that the loaded data contains all required columns.

        Checks that the pairs_dataframe contains the necessary modality 1, modality 2, 
        and label columns as specified in the configuration.

        Raises:
            ValueError: If any required columns are missing from the DataFrame
        """
        # Verify all required columns exist
        required_columns = [self.modality_1_column, self.modality_2_column, self.label_column]
        missing_columns = [col for col in required_columns if col not in self.pairs_dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the dataset into training, validation, and test sets.

        Uses a two-step splitting process:
        1. First splits data into training and a temporary set (containing validation and test data)
        2. Then splits the temporary set into validation and test sets

        The split ratios are determined by the train_val_test_split parameter.

        Args:
            df: DataFrame containing the full dataset

        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames
        """
        # Split the positive pairs into train/val/test
        # First split into train and temp (val+test)
        train_ratio = self.train_val_test_split[0]
        temp_ratio = self.train_val_test_split[1] + self.train_val_test_split[2]

        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            shuffle=True
        )

        # Then split temp into val and test
        val_ratio = self.train_val_test_split[1] / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            shuffle=True
        )
        return train_df, val_df, test_df

    def process_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Process a raw DataFrame by applying sequence length filtering and positive filtering.

        This is a convenience method that applies both reduce_size and filter_positives
        in the correct order to prepare a raw DataFrame for use in the model.

        Args:
            dataframe: Raw DataFrame to process

        Returns:
            Processed DataFrame with sequence length and positive filtering applied
        """
        # filter out the dataframe by sequence length
        processed_df = self.reduce_size(dataframe)
        # checks to see if the dataframe needs to be filtered for positives and if so, filter it
        actual_df = self.filter_positives(processed_df, self.label_column)
        return actual_df

    def create_train_val_and_test(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create or load training, validation, and test datasets.

        This method handles multiple data loading scenarios:
        1. Loading validation and test data from Mochi tasks
        2. Loading validation and test data from separate files
        3. Splitting a single dataset into train/val/test splits

        The method ensures data consistency and handles error cases appropriately.

        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames ready for dataset creation

        Raises:
            ValueError: If incompatible data sources are provided or if pairs_dataframe is None
        """
        # Initialize val_df and test_df as None
        val_df = None
        test_df = None

        # Load validation and test data if provided
        if self.mochi_task is not None and (self.val_file is not None or self.test_file is not None):
            raise ValueError("mochi_task is provided but val_file or test_file were also provided - "
                             "please provide only one of them.")

        if self.mochi_task is not None:
            # mochi data stores train, val and test as three different keys in hdf5 file
            val_df = self.load_mochi_dataframe(
                task_name=self.mochi_task,
                split=VAL_SPLIT,
                task_type=self.mochi_task_type,
                task_version=self.mochi_task_version,
                release_bucket=RELEASE_BUCKET,
                print_version=False
            )
            test_df = self.load_mochi_dataframe(
                task_name=self.mochi_task,
                split=TEST_SPLIT,
                task_type=self.mochi_task_type,
                task_version=self.mochi_task_version,
                release_bucket=RELEASE_BUCKET,
                print_version=False
            )

        elif self.val_file is not None and self.test_file is not None:
            # backward compatibility for loading val and test data from files
            val_df = self.process_dataframe(self.load_dataframe(self.val_file))
            test_df = self.process_dataframe(self.load_dataframe(self.test_file))

        # now, we need to check if we do splits or we have everything. First we check we have the main file

        # pre-process the train dataframe which was already loaded in setup()
        if self.pairs_dataframe is not None:
            train_df = self.pairs_dataframe
            # pre-process the val and test dataframes if they were provided
            if val_df is None and test_df is None:
                # if no val and test dataframes were provided, split the train dataframe into val and test
                # doesn't reduce or filter if the dataframe is directly provided
                train_df, val_df, test_df = self.split_data(train_df)
        else:
            raise ValueError("pairs_dataframe is None.")

        return train_df, val_df, test_df

    def make_dataframe_dataset(
        self,
        df: pd.DataFrame,
        # from_disk: bool = False,
        labels: Optional[list] = None,
        sources: Optional[list[dict]] = None
    ) -> BonbonTokenDataset:
        """Create a BonbonTokenDataset from a DataFrame.

        Extracts modality 1 and modality 2 data from the DataFrame and creates a dataset
        that can be used by the model. Optionally loads pre-calculated embeddings
        from disk if from_disk is True.

        Args:
            df: DataFrame containing modality 1 and modality 2 data
            from_disk: If True, load pre-calculated embeddings from disk paths
                       specified in modality_1_emb_path and modality_2_emb_path
            labels: Optional labels. If None and label column exists in df, uses df labels.
                    If None and no label column, passes None to dataset.
            sources: Optional list of dicts containing source information for each sample.

        Returns:
            BonbonTokenDataset instance ready for use in data loaders
        """
        # if the labels were not provided and we have a label column in the dataframe, use that
        if labels is None and self.label_column in df.columns:
            labels = df[self.label_column].tolist()
        # print(f"===== from_disk is {from_disk} ======")
        # if from_disk:
        #     return BonbonTokenDataset(
        #         modality_1_ids=df[self.modality_1_column].tolist(),
        #         modality_2_ids=df[self.modality_2_column].tolist(),
        #         modality_1_seqs=df[self.modality_1_seq_column].tolist(),
        #         modality_2_seqs=df[self.modality_2_seq_column].tolist(),
        #         labels=labels,
        #         modality_1_type=self.modality_1_type,
        #         modality_2_type=self.modality_2_type,
        #         max_modality_1_seq_length=self.max_modality_1_seq_length,
        #         max_modality_2_seq_length=self.max_modality_2_seq_length,
        #         # loading pre-calculated embeddings or tokenized sequences from disk
        #         modality_1_emb_path=self.modality_1_emb_path,
        #         modality_2_emb_path=self.modality_2_emb_path,
        #         sources=sources,  # Pass sources to dataset
        #     )
        # else:
        #     return BonbonTokenDataset(
        #         modality_1_ids=df[self.modality_1_column].tolist(),
        #         modality_2_ids=df[self.modality_2_column].tolist(),
        #         modality_1_seqs=df[self.modality_1_seq_column].tolist(),
        #         modality_2_seqs=df[self.modality_2_seq_column].tolist(),
        #         labels=labels,
        #         modality_1_type=self.modality_1_type,
        #         modality_2_type=self.modality_2_type,
        #         max_modality_1_seq_length=self.max_modality_1_seq_length,
        #         max_modality_2_seq_length=self.max_modality_2_seq_length,
        #         sources=sources,  # Pass sources to dataset
        #     )
        return BonbonTokenDataset(
            modality_1_ids=df[self.modality_1_column].tolist(),
            modality_2_ids=df[self.modality_2_column].tolist(),
            modality_1_seqs=df[self.modality_1_seq_column].tolist(),
            modality_2_seqs=df[self.modality_2_seq_column].tolist(),
            labels=labels,
            modality_1_type=self.modality_1_type,
            modality_2_type=self.modality_2_type,
            max_modality_1_seq_length=self.max_modality_1_seq_length,
            max_modality_2_seq_length=self.max_modality_2_seq_length,
            # loading pre-calculated embeddings or tokenized sequences from disk
            modality_1_emb_path=self.modality_1_emb_path,
            modality_2_emb_path=self.modality_2_emb_path,
            sources=sources,  # Pass sources to dataset
        )

    def create_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Create BonbonTokenDataset instances for training, validation, and testing.

        Takes the processed DataFrames for each data split and creates corresponding
        dataset objects, storing them as instance attributes for later use in data loaders.
        This method now also processes source information for validation and test sets.

        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            test_df: Test data DataFrame
        """
        # Process sources for each split (only val and test will actually be processed)
        # TODO: fix sources from datasets eg pastel
        # train_sources = self.process_sources(train_df, split='train')
        # val_sources = self.process_sources(val_df, split='val')
        # test_sources = self.process_sources(test_df, split='test')
        #
        # # Store processed source DataFrames for potential debugging/verification
        # self.train_sources = train_sources
        # self.val_sources = val_sources
        # self.test_sources = test_sources

        # Create datasets for each split with sources
        self.train_dataset = self.make_dataframe_dataset(
            train_df,
            # from_disk=self.from_disk,
            # sources=train_sources
        )
        self.val_dataset = self.make_dataframe_dataset(
            val_df,
            # from_disk=self.from_disk,
            # sources=val_sources
        )
        self.test_dataset = self.make_dataframe_dataset(
            test_df,
            # from_disk=self.from_disk,
            # sources=test_sources
        )

        # Log source processing results
        if self.compute_source_metrics and self.unique_sources:
            print(f"Total unique sources found across val/test: {len(self.unique_sources)}")
            print(f"Source categories: {', '.join(self.unique_sources[:10])}"
                  f"{'...' if len(self.unique_sources) > 10 else ''}")

    def load_dataframe(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a tab-separated data file into a pandas DataFrame.

        Reads a TSV file containing modality pairs and their labels,
        with appropriate data types for each column.

        Args:
            file_path: Path to the tab-separated data file

        Returns:
            DataFrame containing the loaded data with appropriate column types
        """
        print(f"Loading pairs from {file_path}")
        return pd.read_csv(
            file_path,
            sep="\t",
            dtype={
                self.modality_1_column: 'string',
                self.modality_2_column: 'string',
                self.label_column: 'float',
            })

    def get_mochi_pairs(
        self,
        task_name: str,
        task_type: str,
        task_version: str,
        release_bucket: str = RELEASE_BUCKET,
        print_version: bool = True
    ) -> Union[str, Path]:
        """Download and retrieve the path to a Mochi dataset file.

        Fetches task information from the specified bucket, downloads the dataset
        to the local NVME folder, and returns the path to the downloaded file.

        Args:
            task_name: Name of the Mochi task
            task_type: Type of the Mochi task
            task_version: Version of the Mochi task (or 'latest')
            release_bucket: S3 bucket containing the Mochi tasks
            print_version: Whether to print version information

        Returns:
            Path to the downloaded dataset file

        Note:
            This method uses the instance's nvme_folder attribute for the download location.
        """
        # Get task information from the bucket
        task_info = get_task(bucket_name=release_bucket,
                             task_name=task_name,
                             task_type=task_type,
                             version=task_version)
        task_version = task_info['version']
        data_file = download_from_s3(task_info['dataset_s3_uri'], self.nvme_folder)
        if print_version:
            print(f"Downloading mochi task {task_name} version {task_version} from S3")
        return data_file

    def load_mochi_dataframe(
        self,
        task_name: str,
        split: str,
        task_type: str,
        task_version: str,
        release_bucket: str = RELEASE_BUCKET,
        print_version: bool = True
    ) -> pd.DataFrame:
        """Load a specific split of a Mochi dataset as a DataFrame with length filtering.

        Downloads the dataset from S3 and applies sequence length filtering based on
        the configured maximum lengths for modality 1 and modality 2 sequences.

        Args:
            task_name: Name of the Mochi task
            split: Dataset split to load ('train', 'val', or 'test')
            task_type: Type of the Mochi task
            task_version: Version of the Mochi task (or 'latest')
            release_bucket: S3 bucket containing the Mochi tasks
            print_version: Whether to print version information

        Returns:
            DataFrame containing the requested split with sequence length filtering applied
        """
        # Get task information and load the data
        task_info = get_task(release_bucket, task_name=task_name, task_type=task_type, version=task_version)
        task_version = task_info['version']
        if print_version:
            print(f"Downloading mochi task {task_name} version {task_version} from S3")
        return load_task_dataset_with_length_filtering(bucket_name=release_bucket,
                                                       task_name=task_name,
                                                       task_type=task_type,
                                                       version=task_version,
                                                       splits=[split],
                                                       storage_directory=self.nvme_folder,
                                                       sequence_filters={self.modality_1_column: self.max_modality_1_seq_length,
                                                                         self.modality_2_column: self.max_modality_2_seq_length},
                                                       label_column=self.label_column,
                                                       filter_positives=self.pos_filtering)[split]

    def prepare_data(self):
        """
        Internal Lightning DataModule method runs only on Rank 0.
        Warning: you cannot set variables with this function as the other ranks would start at the same time.
        Can only end with a disk-writing operation.

        Download the data from S3 into local NVME
        For the mochi parquet implementation: download all splits
        """

        if self.mochi_task is not None:
            start_time = time.time()
            print(f"prepare_data: Downloading mochi task {self.mochi_task} from S3")
            task_info = get_task(RELEASE_BUCKET,
                                 task_name=self.mochi_task,
                                 task_type=self.mochi_task_type,
                                 version=self.mochi_task_version)
            task_version = task_info['version']
            # download all the splits together; really what this does is download the parquet file to cache it
            download_from_s3(task_info['dataset_s3_uri'], self.nvme_folder)
            end_time = time.time()
            print(f"prepare_data: Downloaded mochi task {self.mochi_task} version {task_version} "
                  f"from S3 to {self.nvme_folder} in {end_time - start_time} seconds")

    # TODO: use string literal typing to validate parameter
    def setup(self, stage: Optional[str] = None):
        """
        Set up data splits. Called by Lightning.

        :param stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # TODO: add validation of stage parameter
        if stage in ('fit', None):
            # Download the data from S3 into local NVME

            # check conflicting inputs and raise if both are provided
            if self.mochi_task is not None and self.pairs_file is not None:
                    raise ValueError("Both mochi_task and pairs_file are both provided. Please provide only one of them.")

            # now we should have only mochi_task or pairs_file
            if self.mochi_task is not None:
                self.pairs_dataframe = self.load_mochi_dataframe(task_name=self.mochi_task,
                                                                 split=TRAIN_SPLIT,
                                                                 task_type=self.mochi_task_type,
                                                                 task_version=self.mochi_task_version,
                                                                 release_bucket=RELEASE_BUCKET,
                                                                 print_version=False)

                # TODO: remove deprecated code
                # self.pairs_file = self.get_mochi_pairs(self.mochi_task, self.mochi_task_type, self.mochi_task_version, RELEASE_BUCKET)

            elif self.pairs_file is not None:
                # a pair file or a dataframe was provided, first figure out what type we got
                self.pairs_dataframe = self.process_dataframe(self.load_dataframe(self.pairs_file))

            else:
                raise ValueError("pairs_file or mochi_task must be provided")

            # Verifies columns are in the data
            self.verify_data()
            # split or load val and test
            train_df, val_df, test_df = self.create_train_val_and_test()

            if self.normalize_label:
                train_df, val_df, test_df = self.normalize(train_df, val_df, test_df)

            self.create_datasets(train_df, val_df, test_df)

        # TODO Integrate parquet
        elif stage in ('test', 'predict'):
            # if we're in test mode, all the input dataframe goes to the testing, without any splitting
            # check if we got a dataframe or a file
            print("in test stage")

            # Update: look for test_file instead of pairs_file
            if self.mochi_task is not None and self.test_file is not None:
                raise ValueError("Both mochi_task and test_file are both provided. Please provide only one of them.")

            if self.mochi_task is not None:
                test_df = self.load_mochi_dataframe(task_name=self.mochi_task,
                                                    split=TEST_SPLIT,
                                                    task_type=self.mochi_task_type,
                                                    task_version=self.mochi_task_version,
                                                    release_bucket=RELEASE_BUCKET,
                                                    print_version=False)
            # Update: look for test_file instead of pairs_file
            elif self.test_file is not None:
                # when we are in test stage, the pairs_file is the test file
                test_df = self.load_dataframe(self.test_file)
                test_df = self.reduce_size(test_df)
            else:
                raise ValueError("test_file or mochi_task must be provided")

            if stage == 'predict':
                # For prediction, check if labels exist in the dataframe
                has_labels = self.label_column in test_df.columns

                if not has_labels:
                    print(f"No label column '{self.label_column}' found in prediction data. Using None for labels.")
                    labels = None
                else:
                    labels = test_df[self.label_column].tolist()
                # Process sources for predict (which uses test dataset)
                # TODO fix datasets with sources error eg pastel
                # test_sources = self.process_sources(test_df, split='test')
                # self.test_sources = test_sources

                self.test_dataset = self.make_dataframe_dataset(test_df,
                                                                # from_disk=self.from_disk,
                                                                labels=labels,
                                                                # sources=test_sources
                                                                )
            else:
                # Test mode - labels required
                # test_sources = self.process_sources(test_df, split='test')
                # self.test_sources = test_sources
                # support for loading pre-calculated embeddings or tokenized sequences from disk
                self.test_dataset = self.make_dataframe_dataset(test_df,
                                                                # from_disk=self.from_disk,
                                                                # sources=test_sources
                                                                )

        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """Create the train dataloader.

        Lightning hook: Required method to return a DataLoader for the training data.

        Returns:
            DataLoader configured for training data with shuffling enabled
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader.

        Lightning hook: Required method to return a DataLoader for the validation data.

        Returns:
            DataLoader configured for validation data
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader.

        Lightning hook: Required method to return a DataLoader for the test data.

        Returns:
            DataLoader configured for test data
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,

        )

    def predict_dataloader(self) -> DataLoader:
        """Create the prediction dataloader.

        Lightning hook: Required method to return a DataLoader for prediction.

        Returns:
            DataLoader configured for prediction (using test dataset)
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        )


# TODO: This should be refactored to inherit from BonbonTokenDataModule
# TODO: The docstrings in this need to be fixed
class BonbonMLMDataModule(L.LightningDataModule):
    """Lightning DataModule for handling modality pair datasets.

    This DataModule manages the loading, processing, and batching of modality pair
    data for training, validation, and testing. It supports multiple data sources:

    1. Loading from Mochi tasks (standardized datasets from S3)
    2. Loading from local tab-separated files
    3. Splitting a single dataset into train/val/test partitions

    The module handles sequence length filtering, positive pair filtering, and
    normalization of labels. It creates appropriate PyTorch DataLoaders with
    custom collation functions for efficient batching.

    Attributes:
        pairs_file: Path to the main data file containing modality pairs
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        train_dataset: Dataset object for training data
        val_dataset: Dataset object for validation data
        test_dataset: Dataset object for test data
    """
    def __init__(
            self,

            modality_column: Literal['Protein', 'Molecule'],
            modality_seq_column: Literal['Sequence', 'SELFIES', 'SMILES'],
            modality_type: Literal['protein', 'molecule'],
            max_modality_seq_length: int,
            pairs_file: Union[str, Path] = None,
            batch_size: int = 32,
            num_workers: int = 4,
            train_val_test_split: tuple[float, float, float] = (0.7, 0.15, 0.15),  # TODO redundant
            val_file: Union[str, Path] = None,
            test_file: Union[str, Path] = None,
            persistent_workers: bool = False,
            pos_filtering: bool = False,
            mochi_task: str = None,
            mochi_task_type: Literal['task','evaluation_task'] = 'task',
            mochi_task_version: str = 'latest',
            nvme_folder: Union[str, Path] = '/opt/dlami/nvme/',
    ):
        """
        Initialize the DataModule.

        :param pairs_file: a dataframe of modality pairs with labels
        :param modality_1_column: name of the first modality column in the dataframe
        :param modality_2_column: name of the second modality column in the dataframe
        :param label_column: name of the label column in the dataframe
        :param modality_1_tokenizer_path: Path to first modality tokenizer
        :param modality_2_tokenizer_path: Path to second modality tokenizer
        :param batch_size: Batch size for dataloaders
        :param num_workers: Number of workers for dataloaders
        :param train_val_test_split: Ratios for splitting data
        """
        super().__init__()
        print(f"DEBUG modality_type is set to {modality_type}")
        self.modality_column = modality_column
        self.modality_seq_column = modality_seq_column
        self.modality_type = modality_type
        self.max_modality_seq_length = max_modality_seq_length
        self.pairs_file = pairs_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.val_file = val_file
        self.test_file = test_file
        self.pos_filtering = pos_filtering
        self.persistent_workers = persistent_workers

        self.mochi_task = mochi_task
        self.mochi_task_type = mochi_task_type
        self.mochi_task_version = mochi_task_version
        self.nvme_folder = nvme_folder

        # apply drop last not as a parameter while we are testing the idea
        self.drop_last = True

        # custom collate function
        self.collate_fn = collate_bonbonmlm


        self.save_hyperparameters()
        # initialize dataframes
        self.pairs_dataframe = None # dataframe of main training date
        # initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None


    def reduce_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the DataFrame to include only sequences within the maximum length limits.

        Removes rows where either modality 1 or modality 2 sequences exceed the maximum
        allowed lengths specified in the configuration.

        Args:
            df: DataFrame containing modality 1 and modality 2 sequences

        Returns:
            Filtered DataFrame containing only sequences within length limits

        Raises:
            ValueError: If required sequence columns are missing from the DataFrame
        """
        self.seq_smile_columns = {self.modality_seq_column}  # Add this line

        if self.seq_smile_columns.issubset(df.columns):
            condition = df[self.modality_seq_column].str.len() < self.max_modality_seq_length
            return df[condition]
        else:
            raise ValueError(f"Missing required columns: {self.seq_smile_columns}. Dataframe columns are: {df.columns}")

    def verify_data(self) -> None:
        """Verify that the loaded data contains all required columns.

        Checks that the pairs_dataframe contains the necessary modality 1, modality 2,
        and label columns as specified in the configuration.

        Raises:
            ValueError: If any required columns are missing from the DataFrame
        """
        # Verify all required columns exist
        required_columns = [self.modality_column, self.modality_seq_column]
        missing_columns = [col for col in required_columns if col not in self.pairs_dataframe.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split the dataset into training, validation, and test sets.

        Uses a two-step splitting process:
        1. First splits data into training and a temporary set (containing validation and test data)
        2. Then splits the temporary set into validation and test sets

        The split ratios are determined by the train_val_test_split parameter.

        Args:
            df: DataFrame containing the full dataset

        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames
        """
        # Split the positive pairs into train/val/test
        # First split into train and temp (val+test)
        train_ratio = self.train_val_test_split[0]
        temp_ratio = self.train_val_test_split[1] + self.train_val_test_split[2]

        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            shuffle=True
        )

        # Then split temp into val and test
        val_ratio = self.train_val_test_split[1] / temp_ratio
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            shuffle=True
        )
        return train_df, val_df, test_df

    def process_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Process a raw DataFrame by applying sequence length filtering and positive filtering.

        This is a convenience method that applies both reduce_size and filter_positives
        in the correct order to prepare a raw DataFrame for use in the model.

        Args:
            dataframe: Raw DataFrame to process

        Returns:
            Processed DataFrame with sequence length and positive filtering applied
        """
        # filter out the dataframe by sequence length
        return self.reduce_size(dataframe)

    def create_train_val_and_test(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create or load training, validation, and test datasets.

        This method handles multiple data loading scenarios:
        1. Loading validation and test data from Mochi tasks
        2. Loading validation and test data from separate files
        3. Splitting a single dataset into train/val/test splits

        The method ensures data consistency and handles error cases appropriately.

        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames ready for dataset creation

        Raises:
            ValueError: If incompatible data sources are provided or if pairs_dataframe is None
        """
        # Initialize val_df and test_df as None
        val_df = None
        test_df = None

        # Load validation and test data if provided
        if self.mochi_task is not None and (self.val_file is not None or self.test_file is not None):
            raise ValueError("mochi_task is provided but val_file or test_file were also provided - "
                             "please provide only one of them.")

        if self.mochi_task is not None:
            # mochi data stores train, val and test as three different keys in hdf5 file
            val_df = self.load_mochi_dataframe(
                task_name=self.mochi_task,
                split=VAL_SPLIT,
                task_type=self.mochi_task_type,
                task_version=self.mochi_task_version,
                release_bucket=RELEASE_BUCKET,
                print_version=False
            )
            test_df = self.load_mochi_dataframe(
                task_name=self.mochi_task,
                split=TEST_SPLIT,
                task_type=self.mochi_task_type,
                task_version=self.mochi_task_version,
                release_bucket=RELEASE_BUCKET,
                print_version=False
            )

        elif self.val_file is not None and self.test_file is not None:
            # backward compatibility for loading val and test data from files
            val_df = self.process_dataframe(self.load_dataframe(self.val_file))
            test_df = self.process_dataframe(self.load_dataframe(self.test_file))

        # now, we need to check if we do splits or we have everything. First we check we have the main file

        # pre-process the train dataframe which was already loaded in setup()
        if self.pairs_dataframe is not None:
            train_df = self.pairs_dataframe
            # pre-process the val and test dataframes if they were provided
            if val_df is None and test_df is None:
                # if no val and test dataframes were provided, split the train dataframe into val and test
                # doesn't reduce or filter if the dataframe is directly provided
                train_df, val_df, test_df = self.split_data(train_df)
        else:
            raise ValueError("pairs_dataframe is None.")

        return train_df, val_df, test_df

    def make_dataframe_dataset(
        self,
        df: pd.DataFrame,
    ) -> BonbonMLMDataset:
        """Create a BonbonTokenDataset from a DataFrame.

        Extracts modality 1 and modality 2 data from the DataFrame and creates a dataset
        that can be used by the model. Optionally loads pre-calculated embeddings
        from disk if from_disk is True.

        Args:
            df: DataFrame containing modality 1 and modality 2 data

        Returns:
            BonbonTokenDataset instance ready for use in data loaders
        """

        return BonbonMLMDataset(
            modality_seqs=df[self.modality_seq_column].tolist(),
            modality_type=self.modality_type,
            max_modality_seq_length=self.max_modality_seq_length,
        )


    def create_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Create BonbonTokenDataset instances for training, validation, and testing.

        Takes the processed DataFrames for each data split and creates corresponding
        dataset objects, storing them as instance attributes for later use in data loaders.
        This method now also processes source information for validation and test sets.

        Args:
            train_df: Training data DataFrame
            val_df: Validation data DataFrame
            test_df: Test data DataFrame
        """



        # Create datasets for each split with sources
        self.train_dataset = self.make_dataframe_dataset(train_df)
        self.val_dataset = self.make_dataframe_dataset(val_df)
        self.test_dataset = self.make_dataframe_dataset(test_df)

    def load_dataframe(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load a tab-separated data file into a pandas DataFrame.

        Reads a TSV file containing modality pairs and their labels,
        with appropriate data types for each column.

        Args:
            file_path: Path to the tab-separated data file

        Returns:
            DataFrame containing the loaded data with appropriate column types
        """
        print(f"Loading pairs from {file_path}")
        return pd.read_csv(
            file_path,
            sep="\t",
            dtype={
                self.modality_column: 'string',
                self.modality_seq_column: 'string',
            })

    def get_mochi_pairs(self,
                        task_name: str,
                        task_type: str,
                        task_version: str,
                        release_bucket: str = RELEASE_BUCKET,
                        print_version: bool = True) -> Union[str, Path]:
        """Download and retrieve the path to a Mochi dataset file.

        Fetches task information from the specified bucket, downloads the dataset
        to the local NVME folder, and returns the path to the downloaded file.

        Args:
            task_name: Name of the Mochi task
            task_type: Type of the Mochi task
            task_version: Version of the Mochi task (or 'latest')
            release_bucket: S3 bucket containing the Mochi tasks
            print_version: Whether to print version information

        Returns:
            Path to the downloaded dataset file

        Note:
            This method uses the instance's nvme_folder attribute for the download location.
        """
        # Get task information from the bucket
        task_info = get_task(bucket_name=release_bucket,
                             task_name=task_name,
                             task_type=task_type,
                             version=task_version)
        task_version = task_info['version']
        data_file = download_from_s3(task_info['dataset_s3_uri'], self.nvme_folder)
        if print_version:
            print(f"Downloading mochi task {task_name} version {task_version} from S3")
        return data_file

    def load_mochi_dataframe(self,
                             task_name: str,
                             split: str,
                             task_type: str,
                             task_version: str,
                             release_bucket: str = RELEASE_BUCKET,
                             print_version: bool = True) -> pd.DataFrame:
        """Load a specific split of a Mochi dataset as a DataFrame with length filtering.

        Downloads the dataset from S3 and applies sequence length filtering based on
        the configured maximum lengths for modality 1 and modality 2 sequences.

        Args:
            task_name: Name of the Mochi task
            split: Dataset split to load ('train', 'val', or 'test')
            task_type: Type of the Mochi task
            task_version: Version of the Mochi task (or 'latest')
            release_bucket: S3 bucket containing the Mochi tasks
            print_version: Whether to print version information

        Returns:
            DataFrame containing the requested split with sequence length filtering applied
        """
        # Get task information and load the data
        task_info = get_task(release_bucket, task_name=task_name, task_type=task_type, version=task_version)
        task_version = task_info['version']
        if print_version:
            print(f"Downloading mochi task {task_name} version {task_version} from S3")
        return load_task_dataset_with_length_filtering(bucket_name=release_bucket,
                                                       task_name=task_name,
                                                       task_type=task_type,
                                                       version=task_version,
                                                       splits=[split],
                                                       storage_directory=self.nvme_folder,
                                                       sequence_filters={self.modality_column: self.max_modality_seq_length},
                                                       label_column=None,
                                                       filter_positives=self.pos_filtering)[split]

    def prepare_data(self):
        """
        Internal Lightning DataModule method runs only on Rank 0.
        Warning: you cannot set variables with this function as the other ranks would start at the same time.
        Can only end with a disk-writing operation.

        Download the data from S3 into local NVME
        For the mochi parquet implementation: download all splits
        """

        if self.mochi_task is not None:
            start_time = time.time()
            print(f"prepare_data: Downloading mochi task {self.mochi_task} from S3")
            task_info = get_task(RELEASE_BUCKET,
                                 task_name=self.mochi_task,
                                 task_type=self.mochi_task_type,
                                 version=self.mochi_task_version)
            task_version = task_info['version']
            # download all the splits together; really what this does is download the parquet file to cache it
            download_from_s3(task_info['dataset_s3_uri'], self.nvme_folder)
            end_time = time.time()
            print(f"prepare_data: Downloaded mochi task {self.mochi_task} version {task_version} "
                  f"from S3 to {self.nvme_folder} in {end_time - start_time} seconds")

    # TODO: use string literal typing to validate parameter
    def setup(self, stage: Optional[str] = None):
        """
        Set up data splits. Called by Lightning.

        :param stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # TODO: add validation of stage parameter
        if stage in ('fit', None):
            # Download the data from S3 into local NVME

            # check conflicting inputs and raise if both are provided
            if self.mochi_task is not None and self.pairs_file is not None:
                    raise ValueError("Both mochi_task and pairs_file are both provided. Please provide only one of them.")

            # now we should have only mochi_task or pairs_file
            if self.mochi_task is not None:
                self.pairs_dataframe = self.load_mochi_dataframe(task_name=self.mochi_task,
                                                                 split=TRAIN_SPLIT,
                                                                 task_type=self.mochi_task_type,
                                                                 task_version=self.mochi_task_version,
                                                                 release_bucket=RELEASE_BUCKET,
                                                                 print_version=False)

                # TODO: remove deprecated code
                # self.pairs_file = self.get_mochi_pairs(self.mochi_task, self.mochi_task_type, self.mochi_task_version, RELEASE_BUCKET)

            elif self.pairs_file is not None:
                # a pair file or a dataframe was provided, first figure out what type we got
                self.pairs_dataframe = self.process_dataframe(self.load_dataframe(self.pairs_file))

            else:
                raise ValueError("pairs_file or mochi_task must be provided")

            # Verifies columns are in the data
            self.verify_data()
            # split or load val and test
            train_df, val_df, test_df = self.create_train_val_and_test()

            self.create_datasets(train_df, val_df, test_df)

        # TODO Integrate parquet
        elif stage in ('test', 'predict'):
            # if we're in test mode, all the input dataframe goes to the testing, without any splitting
            # check if we got a dataframe or a file
            print("in test stage")

            # Update: look for test_file instead of pairs_file
            if self.mochi_task is not None and self.test_file is not None:
                raise ValueError("Both mochi_task and test_file are both provided. Please provide only one of them.")

            if self.mochi_task is not None:
                test_df = self.load_mochi_dataframe(task_name=self.mochi_task,
                                                    split=TEST_SPLIT,
                                                    task_type=self.mochi_task_type,
                                                    task_version=self.mochi_task_version,
                                                    release_bucket=RELEASE_BUCKET,
                                                    print_version=False)
            # Update: look for test_file instead of pairs_file
            elif self.test_file is not None:
                # when we are in test stage, the pairs_file is the test file
                test_df = self.load_dataframe(self.test_file)
                test_df = self.reduce_size(test_df)
            else:
                raise ValueError("test_file or mochi_task must be provided")

            if stage == 'predict':
                # For prediction, check if labels exist in the dataframe
                has_labels = self.label_column in test_df.columns

                if not has_labels:
                    print(f"No label column '{self.label_column}' found in prediction data. Using None for labels.")
                    labels = None
                else:
                    labels = test_df[self.label_column].tolist()
                # Process sources for predict (which uses test dataset)
                # test_sources = self.process_sources(test_df, split='test')
                # self.test_sources = test_sources

                self.test_dataset = self.make_dataframe_dataset(test_df,
                                                                # from_disk=self.from_disk,
                                                                labels=labels,
                                                                #sources=test_sources
                                                                )
            else:
                # Test mode - labels required
                # test_sources = self.process_sources(test_df, split='test')
                # self.test_sources = test_sources
                # support for loading pre-calculated embeddings or tokenized sequences from disk
                self.test_dataset = self.make_dataframe_dataset(test_df,
                                                                # from_disk=self.from_disk,
                                                                #sources=test_sources
                                                                )

        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """Create the train dataloader.

        Lightning hook: Required method to return a DataLoader for the training data.

        Returns:
            DataLoader configured for training data with shuffling enabled
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last
        )

    def val_dataloader(self) -> DataLoader:
        """Create the validation dataloader.

        Lightning hook: Required method to return a DataLoader for the validation data.

        Returns:
            DataLoader configured for validation data
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last
        )

    def test_dataloader(self) -> DataLoader:
        """Create the test dataloader.

        Lightning hook: Required method to return a DataLoader for the test data.

        Returns:
            DataLoader configured for test data
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers,

        )

    def predict_dataloader(self) -> DataLoader:
        """Create the prediction dataloader.

        Lightning hook: Required method to return a DataLoader for prediction.

        Returns:
            DataLoader configured for prediction (using test dataset)
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers
        )
