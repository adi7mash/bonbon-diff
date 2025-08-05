import re
import logging
from typing import Union, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from bonbontoken.tokenizer import MODALITY_SETTINGS
from bonbontoken.utils import resolve_tokenizer_path


def _preprocess_sequence_by_type(sequence: str, modality_type: str) -> str:
    """Preprocess a sequence based on its modality type.

    This method handles different preprocessing requirements for different modalities.
    For proteins, it replaces special amino acids and ensures proper formatting.
    For molecules, it introduces spaces between SELFIES tokens.

    Args:
        sequence: The sequence to preprocess.
        modality_type: The type of modality (e.g., 'protein', 'molecule').

    Returns:
        The preprocessed sequence as a string.
    """
    if modality_type in ['protein', 'esm2', 'protbert']:
        # protein specific preprocessing
        try:
            sequence = re.sub(r"[UZOB]", "X", sequence)
        except:
            raise ValueError(f"Invalid sequence: {sequence}")
        # Ensure uppercase for all sequences
        sequence = sequence.upper()
        # ProtBert requires spaces between residues
        return ' '.join(sequence)
    elif modality_type in ['molecule', 'selfiested']:
        # molecule-specific preprocessing, adding spaces between SELFIES tokens
        return sequence.replace('][', '] [')
    elif modality_type == 'chemberta':
        return sequence
    else:
        return sequence  # No preprocessing for other types for future extensibility


class BonbonTokenDataset(Dataset):
    """A PyTorch Dataset for modality pairs.

    This dataset handles pairs of modalities (e.g., protein-molecule pairs) and provides
    functionality for tokenization, embedding loading, and batch preparation. It supports
    various modality types and can handle optional labels for supervised learning tasks.

    The dataset automatically loads appropriate tokenizers based on the specified modality
    types and handles preprocessing of sequences according to modality-specific requirements.

    Attributes:
        modality_1_ids: List of modality 1 identifiers
        modality_2_ids: List of modality 2 identifiers
        modality_1_seqs: List of modality 1 sequences
        modality_2_seqs: List of modality 2 sequences
        modality_1_type: Type of modality 1 (e.g., 'protein')
        modality_2_type: Type of modality 2 (e.g., 'molecule',)
        max_modality_1_seq_length: Maximum length for modality 1 sequences
        max_modality_2_seq_length: Maximum length for modality 2 sequences
        modality_1_emb_path: Path to modality 1 embeddings folder
        modality_2_emb_path: Path to modality 2 embeddings folder
        labels: List of labels for each pair
        sources: List of dictionaries containing source information for each sample
        modality_1_tokenizer: Tokenizer for modality 1
        modality_2_tokenizer: Tokenizer for modality 2
    """
    def __init__(
        self,
        modality_1_ids: list[str],
        modality_2_ids: list[str],
        modality_1_seqs: list[str],
        modality_2_seqs: list[str],
        modality_1_type: str,
        modality_2_type: str,
        max_modality_1_seq_length: int,
        max_modality_2_seq_length: int,
        labels: Optional[Union[list[int], list[float], list[list[int]], list[list[float]]]] = None,
        modality_1_emb_path: Optional[Union[str, Path]] = None,
        modality_2_emb_path: Optional[Union[str, Path]] = None,
        sources: Optional[list[dict]] = None

    ):
        """Initialize the BonbonTokenDataset.

        Args:
            modality_1_ids: List of modality 1 identifiers (e.g., protein IDs)
            modality_2_ids: List of modality 2 identifiers (e.g., molecule IDs)
            modality_1_seqs: List of modality 1 sequences (e.g., protein sequences)
            modality_2_seqs: List of modality 2 sequences (e.g., SELFIES strings)
            modality_1_type: Type of modality 1 (e.g., 'protein')
            modality_2_type: Type of modality 2 (e.g., 'molecule')
            max_modality_1_seq_length: Maximum length for modality 1 sequences
            max_modality_2_seq_length: Maximum length for modality 2 sequences
            labels: Optional list of labels for each pair. If None, dummy labels are used.
            modality_1_emb_path: Path to modality 1 embeddings folder
            modality_2_emb_path: Path to modality 2 embeddings folder
            sources: Optional list of dictionaries containing source information for each sample.

        Raises:
            ValueError: If lengths of modality_1_ids and modality_2_ids don't match
            ValueError: If length of sources doesn't match length of modality ids
        """
        if len(modality_1_ids) != len(modality_2_ids):
            raise ValueError("Length of modality_1_ids and modality_2_ids must match")
        if sources is not None and len(sources) != len(modality_1_ids):
            raise ValueError(
                f"Length of sources ({len(sources)}) must match length of modality ids ({len(modality_1_ids)})")

        # TODO: Add validation for modality types (check if in MODALITY_SETTINGS)

        if sources is None:
            # Create empty dicts for each sample to maintain consistency
            self.sources = [{} for _ in range(len(modality_1_ids))]
        else:
            self.sources = sources

        self.modality_1_ids = modality_1_ids
        self.modality_2_ids = modality_2_ids
        self.modality_1_seqs = modality_1_seqs
        self.modality_2_seqs = modality_2_seqs
        self.modality_1_type = modality_1_type
        self.modality_2_type = modality_2_type
        self.max_modality_1_seq_length = max_modality_1_seq_length
        self.max_modality_2_seq_length = max_modality_2_seq_length

        if modality_1_emb_path is not None:
            self.modality_1_emb_path = Path(modality_1_emb_path)
        else:
            self.modality_1_emb_path = None

        if modality_2_emb_path is not None:
            self.modality_2_emb_path = Path(modality_2_emb_path)
        else:
            self.modality_2_emb_path = None

        # Create int ids instead of str to be able to turn ids list into tensor
        modality_1_le = LabelEncoder()
        modality_2_le = LabelEncoder()
        self.modality_1_ids_int = modality_1_le.fit_transform(self.modality_1_ids)
        self.modality_2_ids_int = modality_2_le.fit_transform(self.modality_2_ids)

        # Handle optional labels
        if labels is None:
            # Create dummy labels for prediction
            self.labels = [-1] * len(modality_1_ids)  # Use -1 as a sentinel value
        else:
            self.labels = labels

        self.modality_1_tokenizer_path = Path(MODALITY_SETTINGS[self.modality_1_type]['tokenizer_path'])
        self.modality_2_tokenizer_path = Path(MODALITY_SETTINGS[self.modality_2_type]['tokenizer_path'])

        # Validate paths - calls resolve_tokenizer_path that checks if it is a path or a hugging face hub id
        # if neither raises an error
        self.modality_1_tokenizer_path = resolve_tokenizer_path(self.modality_1_tokenizer_path)
        self.modality_2_tokenizer_path = resolve_tokenizer_path(self.modality_2_tokenizer_path)

        # load tokenizers
        self.modality_1_tokenizer = AutoTokenizer.from_pretrained(self.modality_1_tokenizer_path)
        self.modality_2_tokenizer = AutoTokenizer.from_pretrained(self.modality_2_tokenizer_path)

        self.logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        """Return the number of modality pairs."""
        # TODO: len for HD5 file should different
        #        self.length = len(self.data_group['modality_1_ids'])

        return len(self.modality_1_ids)

    def _tokenize_sequence(self, sequence: str, is_modality_1: bool, max_seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a sequence using the appropriate tokenizer.

        Both modalities now use the same tokenization parameters for consistency.
        Special tokens (like [CLS], [SEP]) are added by default for both.

        This method handles the tokenization process for both modality types, selecting
        the appropriate tokenizer based on the is_modality_1 flag. It processes the input
        sequence, adds special tokens, and returns the valid tokens along with an attention
        mask indicating which tokens should be attended to.

        Args:
            sequence: The input sequence to tokenize
            is_modality_1: Boolean flag indicating whether to use modality 1 tokenizer (True)
                          or modality 2 tokenizer (False)
            max_seq: Maximum sequence length for tokenization

        Returns:
            A tuple containing:
            - valid_tokens: Tensor of token IDs [seq_len]
            - attention_mask: Boolean tensor indicating valid tokens [seq_len]
              (all True since padding is handled in the collate function)

        Raises:
            Exception: If tokenization fails for any reason
        """
        # select the appropriate tokenizer based on modality
        tokenizer = self.modality_1_tokenizer if is_modality_1 else self.modality_2_tokenizer
        modality_name = "modality 1" if is_modality_1 else "modality 2"

        try:
            encoding = tokenizer(
                sequence,
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq,
                return_tensors='pt',
            )
            # we're not specifying here padding='max_length' to save memory, and
            # we expect an attention mask of only ones

        except Exception as e:
            self.logger.error(f"Error loading tokenizing {modality_name} sequence {sequence}: {str(e)}")
            raise

        # explicitly extract the valid tokens and return valid tokens (de-masked) and the attention mask
        # the attention mask is used in the collate function for batch padding
        # extract tokens
        token_ids = encoding['input_ids'].squeeze(0)

        # get mask and convert mask to bool
        attention_input_mask = encoding['attention_mask'].squeeze(0).bool()
        # extract valid tokens
        valid_tokens = token_ids[attention_input_mask]
        # create a mask in the length of the valid tokens with only ones
        attention_mask = torch.ones(valid_tokens.size(0), dtype=torch.bool)

        # return both values
        # return {'input_ids': valid_tokens, 'attention_mask': attention_mask}

        return valid_tokens, attention_mask

    # TODO complete docstring
    def _load_embedding(self, embedding_id: str, is_modality_1: bool) -> torch.Tensor:
        """Load a single embedding from file."""
        path = (self.modality_1_emb_path if is_modality_1 else self.modality_2_emb_path) / f"{embedding_id}.pt"

        try:
            return torch.load(path, weights_only=True)
        except Exception as e:
            self.logger.error(f"Error loading embedding {embedding_id}: {str(e)}")
            raise

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, str, str, dict]:
        """Get a modality pair embedding and their attention masks.

        This method retrieves data for a single sample at the given index, including
        embeddings for both modalities, their attention masks, labels, identifiers,
        and source information.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            A tuple containing ten elements:
            - modality_1_emb: Tensor of token IDs or embeddings for modality 1
              [seq_len_1] or [seq_len_1, hidden_dim]
            - modality_2_emb: Tensor of token IDs or embeddings for modality 2
              [seq_len_2] or [seq_len_2, hidden_dim]
            - modality_1_mask: Boolean attention mask for modality 1 [seq_len_1],
              where True indicates valid tokens to attend to
            - modality_2_mask: Boolean attention mask for modality 2 [seq_len_2],
              where True indicates valid tokens to attend to
            - label_tensor: Float tensor containing the label(s) for this pair [label_dim]
            - modality_1_id_int: Integer tensor containing the numerical ID for modality 1 [1]
            - modality_2_id_int: Integer tensor containing the numerical ID for modality 2 [1]
            - modality_1_id: String identifier for modality 1 (e.g., protein ID)
            - modality_2_id: String identifier for modality 2 (e.g., molecule ID)
            - source_dict: Dictionary containing source information for this sample

        Raises:
            ValueError: If any embeddings contain NaN or Inf values, or if embeddings are all zeros
        """

        # Get source dict for this index
        modality_1_id = self.modality_1_ids[idx]
        modality_2_id = self.modality_2_ids[idx]

        modality_1_id_int = torch.tensor(self.modality_1_ids_int[idx], dtype=torch.int)
        modality_2_id_int = torch.tensor(self.modality_2_ids_int[idx], dtype=torch.int)

        labels = self.labels[idx]

        label_tensor = torch.tensor(labels, dtype=torch.float)

        # either load or tokenize
        # This code allows us to use either precomputed embeddings or tokenize on the fly.
        if self.modality_1_emb_path is not None:
            # Load embeddings
            modality_1_emb = self._load_embedding(modality_1_id, is_modality_1=True)
            if torch.isnan(modality_1_emb).any() or torch.isinf(modality_1_emb).any():
                raise ValueError(f"{modality_1_id} NaN or Inf found when loading modality 1 embedding")

            # create attention masks for the full sequences
            modality_1_mask = torch.ones(modality_1_emb.size(0), dtype=torch.bool)
        else:
            # tokenize on the fly
            # first pre-process the sequences to match tokenizer requirements
            modality_1_seq = _preprocess_sequence_by_type(
                self.modality_1_seqs[idx],
                self.modality_1_type
            )
            # Tokenize modality 1 and get the masks
            modality_1_emb, modality_1_mask = self._tokenize_sequence(modality_1_seq, True,
                                                                      self.max_modality_1_seq_length)
            if torch.all(modality_1_emb == 0):
                raise ValueError(f"Warning: {modality_1_id} has all-zero embeddings for modality 1")

            if torch.isnan(modality_1_emb).any() or torch.isinf(modality_1_emb).any():
                raise ValueError(f"{modality_1_id} NaN or Inf found in modality 1 tokens")

        if self.modality_2_emb_path is not None:
            modality_2_emb = self._load_embedding(modality_2_id, is_modality_1=False)
            if torch.isnan(modality_2_emb).any() or torch.isinf(modality_2_emb).any():
                raise ValueError(f"{modality_2_id} NaN or Inf found when loading modality 2 embedding")

            # create attention masks for the full sequences
            modality_2_mask = torch.ones(modality_2_emb.size(0), dtype=torch.bool)

        else:
            # tokenize on the fly
            # first pre-process the sequences to match tokenizer requirements
            modality_2_seq = _preprocess_sequence_by_type(
                self.modality_2_seqs[idx],
                self.modality_2_type
            )
            # Tokenize modality 2 and get the masks
            modality_2_emb, modality_2_mask = self._tokenize_sequence(modality_2_seq, False, self.max_modality_2_seq_length)
            if torch.all(modality_2_emb == 0):
                raise ValueError(f"Warning: {modality_2_id} has all-zero embeddings for modality 2")

            if torch.isnan(modality_2_emb).any() or torch.isinf(modality_2_emb).any():
                raise ValueError(f"{modality_2_id} NaN or Inf found in modality 2 tokens")

        source_dict = self.sources[idx]

        return (modality_1_emb, modality_2_emb, modality_1_mask, modality_2_mask, label_tensor,
                modality_1_id_int, modality_2_id_int, modality_1_id, modality_2_id, source_dict)


# TODO consider fixing. Excess cats, list appends, etc
# Custom collate function to handle stacking of variable length sequences
def collate_bonbontoken(
        batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str, dict]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str], list[dict]]:
    """
    Custom collate function for BonbonTokenDataset that handles variable length sequences.
    Padding is done to the within-batch maximum for efficiency.
    """
    # Unzip the batch into separate lists
    (modality_1_input_ids, modality_2_input_ids, modality_1_masks, modality_2_masks, labels,
     modality_1_int_ids, modality_2_int_ids, modality_1_ids, modality_2_ids, source_dicts) = zip(*batch)

    # when unpacking the batch, the first 7 elements are tensors and the last two are tuples

    # Get max lengths in this batch
    max_modality_1_len = max(p.size(0) for p in modality_1_input_ids)
    max_modality_2_len = max(m.size(0) for m in modality_2_input_ids)

    # Pad sequences
    padded_modality_1 = []
    padded_modality_2 = []
    padded_modality_1_masks = []
    padded_modality_2_masks = []

    for m1_emb, m2_emb, m1_mask, m2_mask in zip(modality_1_input_ids, modality_2_input_ids, modality_1_masks, modality_2_masks):
        # Pad modality 1
        m1_padding_len = max_modality_1_len - m1_emb.size(0)
        if m1_padding_len > 0:
            # pad with zeros
            # check if we are using tokenized embeddings or precomputed embeddings
            # where precomputed embeddings are 2D tensors
            if m1_emb.dim() == 1:
                padding = torch.zeros(m1_padding_len, dtype=m1_emb.dtype, device=m1_emb.device)
            else:  # assume 2D tensor for precomputed embeddings
                padding = torch.zeros(m1_padding_len, m1_emb.size(1), dtype=m1_emb.dtype, device=m1_emb.device)
            m1_emb = torch.cat([m1_emb, padding], dim=0)
            m1_mask = torch.cat(
                [m1_mask, torch.zeros(m1_padding_len, dtype=torch.bool, device=m1_mask.device)])

        # Pad modality 2
        m2_padding_len = max_modality_2_len - m2_emb.size(0)
        if m2_padding_len > 0:
            # pad with zeros
            if m2_emb.dim() == 1:
                padding = torch.zeros(m2_padding_len, dtype=m2_emb.dtype, device=m2_emb.device)
            else:  # assume 2D tensor for precomputed embeddings
                padding = torch.zeros(m2_padding_len, m2_emb.size(1), dtype=m2_emb.dtype, device=m2_emb.device)
            m2_emb = torch.cat([m2_emb, padding], dim=0)
            # Extend mask with False for padded positions
            m2_mask = torch.cat(
                [m2_mask, torch.zeros(m2_padding_len, dtype=torch.bool, device=m2_mask.device)])

        padded_modality_1.append(m1_emb)
        padded_modality_2.append(m2_emb)
        padded_modality_1_masks.append(m1_mask)
        padded_modality_2_masks.append(m2_mask)

    # Stack all tensors
    modality_1_tensor = torch.stack(padded_modality_1)
    modality_2_tensor = torch.stack(padded_modality_2)
    modality_1_mask_tensor = torch.stack(padded_modality_1_masks)
    modality_2_mask_tensor = torch.stack(padded_modality_2_masks)
    label_tensor = torch.stack(labels)
    modality_1_int_ids_tensor = torch.stack(modality_1_int_ids)
    modality_2_int_ids_tensor = torch.stack(modality_2_int_ids)
    # TODO: is the below comment false?
    # IDs are not returned as tensors but as lists for readability and ease of debugging
    # Convert IDs to tensors
    # The last three items are converted from tuples to lists
    return (modality_1_tensor, modality_2_tensor, modality_1_mask_tensor, modality_2_mask_tensor,
            label_tensor, modality_1_int_ids_tensor, modality_2_int_ids_tensor,
            list(modality_1_ids), list(modality_2_ids), list(source_dicts))


class BonbonMLMDataset(Dataset):
    # TODO complete docstring to have google styling
    def __init__(
            self,
            modality_seqs: list[str],
            modality_type: str,
            max_modality_seq_length: int,
    ):
        """


        Args:
            modality_seqs: List of modality 1 sequences
            modality_type: Type of the modality (e.g., 'protein', 'molecule')
            max_modality_seq_length: Maximum length for modality 1 sequences
        """

        self.modality_seqs = modality_seqs
        self.modality_type = modality_type
        self.max_modality_seq_length = max_modality_seq_length

        self.tokenizer_path = Path(MODALITY_SETTINGS[self.modality_type]['tokenizer_path'])

        # Validate paths
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {self.tokenizer_path}")

        # load tokenizer
        self.modality_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        self.logger = logging.getLogger(__name__)

    def __len__(self) -> int:
        """Return the number of lines."""

        return len(self.modality_seqs)

    # TODO complete docstring
    def _tokenize_sequence(self, sequence: str, max_seq: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a sequence using the appropriate tokenizer.

        Both modalities now use the same tokenization parameters for consistency.
        Special tokens (like [CLS], [SEP]) are added by default for both.
        """
        # select the appropriate tokenizer based on modality
        tokenizer = self.modality_tokenizer

        try:
            encoding = tokenizer(
                sequence,
                add_special_tokens=True,
                truncation=True,
                max_length=max_seq,
                return_tensors='pt',
            )
            # we're not specifying here padding='max_length' to save memory, and
            # we expect an attention mask of only ones

        except Exception as e:
            self.logger.error(f"Error loading tokenizing sequence {sequence}: {str(e)}")
            raise

        # explicitly extract the valid tokens and return valid tokens (de-masked) and the attention mask
        # the attention mask is used in the collate function for batch padding
        # extract tokens
        token_ids = encoding['input_ids'].squeeze(0)

        # get mask and convert mask to bool
        attention_input_mask = encoding['attention_mask'].squeeze(0).bool()
        # extract valid tokens
        valid_tokens = token_ids[attention_input_mask]
        # create a mask in the length of the valid tokens with only ones
        attention_mask = torch.ones(valid_tokens.size(0), dtype=torch.bool)

        # return both values
        # return {'input_ids': valid_tokens, 'attention_mask': attention_mask}

        return valid_tokens, attention_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a modality pair embedding and their attention masks.

        This method retrieves data for a single sample at the given index, including
        embeddings for both modalities, their attention masks, labels, identifiers,
        and source information.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            A tuple containing ten elements:
            - modality_emb: Tensor of token IDs
              [seq_len_1] or [seq_len_1, hidden_dim]
            - modality_mask: Boolean attention mask [seq_len_1],
              where True indicates valid tokens to attend to

        Raises:
            ValueError: If any embeddings contain NaN or Inf values, or if embeddings are all zeros
        """

        # This code allows us to use either precomputed embeddings or tokenize on the fly.

        # tokenize on the fly
        # first pre-process the sequences to match tokenizer requirements
        modality_seq = _preprocess_sequence_by_type(
            self.modality_seqs[idx],
            self.modality_type
        )
        # Tokenize modality 1 and get the masks
        modality_emb, modality_mask = self._tokenize_sequence(modality_seq, self.max_modality_seq_length)

        if torch.all(modality_emb == 0):
            raise ValueError(f"Warning: has all-zero embeddings {modality_emb}")

        if torch.isnan(modality_emb).any() or torch.isinf(modality_emb).any():
            raise ValueError(f"NaN or Inf found in tokens {modality_emb} ")

        return modality_emb, modality_mask


# TODO consider fixing. Excess cats, list appends, etc
# Custom collate function to handle stacking of variable length sequences
def collate_bonbonmlm(
        batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Custom collate function for BonbonTokenDataset that handles variable length sequences.
    Padding is done to the within-batch maximum for efficiency.
    """
    # Unzip the batch into separate lists
    input_ids, masks = zip(*batch)

    # when unpacking the batch, the first 7 elements are tensors and the last two are tuples

    # Get max lengths in this batch
    max_len = max(s.size(0) for s in input_ids)

    # Pad sequences
    padded_inputs = []
    padded__masks = []

    for s_emb, s_mask in zip(input_ids, masks):
        # Pad modality 1
        padding_len = max_len - s_emb.size(0)
        if padding_len > 0:
            # pad with zeros
            # check if we are using tokenized embeddings or precomputed embeddings
            # where precomputed embeddings are 2D tensors
            if s_emb.dim() == 1:
                padding = torch.zeros(padding_len, dtype=s_emb.dtype, device=s_emb.device)
            else: # assume 2D tensor for precomputed embeddings
                padding = torch.zeros(padding_len, s_emb.size(1), dtype=s_emb.dtype, device=s_emb.device)
            s_emb = torch.cat([s_emb, padding], dim=0)
            s_mask = torch.cat(
                [s_mask, torch.zeros(padding_len, dtype=torch.bool, device=s_mask.device)])

        padded_inputs.append(s_emb)
        padded__masks.append(s_mask)

    # Stack all tensors
    input_ids_tensor = torch.stack(padded_inputs)
    mask_tensor = torch.stack(padded__masks)

    return input_ids_tensor, mask_tensor


# Example usage
if __name__ == "__main__":
    # Sample data
    modality_1_ids = ["protein1", "protein2", "protein3"]
    modality_2_ids = ["molecule1", "molecule2", "molecule3"]
    labels = [0, 1, 1]

    # Create dataset
    dataset = BonbonTokenDataset(
        modality_1_ids=modality_1_ids,
        modality_2_ids=modality_2_ids,
        labels=labels,
        modality_1_emb_path="path/to/modality_1/embeddings",
        modality_2_emb_path="path/to/modality_2/embeddings",
        )

    # Use with DataLoader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Iterate through batches
    for modality_1_batch, modality_2_batch in dataloader:
        # Process batch
        pass
