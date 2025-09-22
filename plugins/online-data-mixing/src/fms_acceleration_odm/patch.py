# fms-hf-tuning patch
from datasets import IterableDataset, Dataset, IterableDatasetDict, DatasetDict
from typing import Union, List
from logging import getLogger
from odm import OnlineData
from tuning.data.setup_dataprocessor import _process_dataconfig_file, _process_raw_data_args, is_pretokenized_dataset
from tuning.data.data_preprocessing_utils import get_data_collator

logger = getLogger(__name__)


def patch_fms_hf_tuning_data_utils_for_odm():
    # Third Party
    # pylint: disable=import-outside-toplevel
    from fms_acceleration.model_patcher import patch_target_module

    patch_target_module("tuning.data.data_processors._process_dataset_configs", _process_dataset_configs)
    patch_target_module("tuning.data.setup_dataprocessor.process_dataargs", process_dataargs)

def _process_dataset_configs(
    self, dataset_configs
) -> Union[Dataset, IterableDataset]:

    splitName = "train"  # default
    all_datasetdicts = {}
    
    logger.info("Starting DataPreProcessor...")
    
    # Now Iterate over the multiple datasets provided to us to process
    for d in dataset_configs:
        logger.info("Loading %s", d.name)

        # In future the streaming etc go as kwargs of this function
        raw_dataset = self.load_dataset(d, self.processor_config.streaming)
        logger.info("Loaded raw dataset : %s", str(raw_dataset))

        if isinstance(raw_dataset, IterableDataset):
            raw_datasets = IterableDatasetDict()
        else:
            raw_datasets = DatasetDict()

        # Assume all is train split
        if isinstance(raw_dataset, (Dataset, IterableDataset)):
            raw_datasets[splitName] = raw_dataset
        else:
            raw_datasets = raw_dataset

        if d.data_handlers:  # Execute the datahandlers
            for data_handler_config in d.data_handlers:
                raw_datasets = self._execute_data_handlers(
                    raw_datasets=raw_datasets,
                    data_handler_config=data_handler_config,
                    splitName=splitName,
                    datasetName=d.name,
                )

        # category --> train split of the dataset
        # always assumed train split is available
        # dataset name in the data config are unique
        # and the name is used as the category or domain name
        assert "train" in raw_datasets
        all_datasetdicts[d.name] = raw_datasets["train"]

    return all_datasetdicts


def process_dataargs(
    data_args,
    tokenizer,
    train_args,
    additional_data_handlers = None,
    is_padding_free = False,
    processor = None,
    is_multipack = False,
):
    max_seq_length = min(train_args.max_seq_length, tokenizer.model_max_length)
    logger.info("Max sequence length is %s", max_seq_length)
    if train_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            "max_seq_length %s exceeds tokenizer.model_max_length \
            %s, using tokenizer.model_max_length %s",
            train_args.max_seq_length,
            tokenizer.model_max_length,
            tokenizer.model_max_length,
        )

    train_dataset = eval_dataset = dataset_text_field = None

    if processor and not (
        data_args.dataset_text_field or data_args.dataset_image_field
    ):
        raise ValueError(
            f"When running a vision model you must provide the dataset_text_field and \
            dataset_image_field for the columns in the dataset. Values should be from \
            column names: {train_dataset.column_names}",
        )

    if data_args.data_config_path:
        train_dataset, eval_dataset, dataset_text_field = _process_dataconfig_file(
            data_args,
            train_args,
            tokenizer,
            additional_data_handlers,
            processor,
            is_multipack,
        )
    else:
        train_dataset, eval_dataset, dataset_text_field = _process_raw_data_args(
            data_args,
            tokenizer,
            train_args.packing,
            max_seq_length,
            additional_data_handlers,
            is_padding_free,
            processor,
        )
    collators = {}
    for k, v in train_dataset:
        is_tokenized_dataset = is_pretokenized_dataset(v)
        data_collator = get_data_collator(
            train_args.packing,
            data_args.response_template,
            tokenizer,
            is_tokenized_dataset,
            max_seq_length,
            data_args.instruction_template,
            is_padding_free=is_padding_free,
            processor=processor,
        )
        collators[k] = data_collator
    print("train_args in patch", train_args)
    train_dataset = OnlineData(train_dataset, collators, train_args.odm_sampling_weights, train_args.odm_gamma, train_args.odm_eta)
    dataset_kwargs = {}
    # For vision model tuning prepare_dataset is skipped.
    if processor is not None:
        dataset_kwargs["skip_prepare_dataset"] = True
    if isinstance(train_dataset, IterableDataset):
        train_args.accelerator_config = {"split_batches": True}
        logger.info(
            "Setting `split_batches` to true - splitting batches among devices \
                    `per_device_train_batch_size` is now the global batch size, and \
                    should be treated as such. The main process will fetch a full \
                    batch and slice it into `num_processes` batches for each process."
        )
    return (
        train_dataset,
        eval_dataset,
        dataset_text_field,
        None,
        max_seq_length,
        dataset_kwargs,
    )