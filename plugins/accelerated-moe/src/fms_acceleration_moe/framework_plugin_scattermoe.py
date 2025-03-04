# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import Dict, Tuple
import asyncio
import os

# Third Party
from accelerate import Accelerator
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch

# Local
from .utils import (
    patch_huggingface_save_and_load_for_dtensors,
    patch_torch_optim_foreach_to_not_apply_to_dtensors,
    prepare_scattermoe,
    recover_safetensors_from_dcp,
)


# pylint: disable=too-many-instance-attributes
class ScatterMoEAccelerationPlugin(AccelerationPlugin):

    # NOTE: we cannot do
    # - require_packages = {"khd"}
    # this is because the khd fork is not properly packaged as a PyPI project, and so
    # - "importlib.util.find_spec('khd')" returns, but
    # - "importlib.metadata.version('kernel-hyperdrive')" does not return
    # if we decide to extract the kernels, then we do not need to anymore,
    # https://github.com/foundation-model-stack/fms-acceleration/issues/105

    restricted_model_archs = [
        "GraniteMoeForCausalLM",
        "MixtralForCausalLM",
        "GraniteMoeSharedForCausalLM",
    ]

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # ep_degree determines the expert parallel sharding
        # - default of 1 means experts are not sharded and operate in pure replication.
        self._ep_degree = self._check_config_and_maybe_check_values(
            key="training.moe.scattermoe.ep_degree",
            default=1,
        )

    @property
    def requires_augmentation(self):
        return True

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        rank, world_size = 0, 1
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            # we do not need to use the fallback as this is wrapped in an `is_initialized` block
            rank = torch.distributed.get_node_local_rank()

        if not hasattr(model.config, "name_or_path") or not model.config.name_or_path:
            raise ValueError(
                "The model configuration is missing the 'name_or_path' attribute."
            )

        model_name = model.config.name_or_path

        self._moe_component_module_names = prepare_scattermoe(
            model,
            checkpoint_name_or_path=model_name,
            rank=rank,
            world_size=world_size,
            ep_degree=self._ep_degree,
            mixed_precision=False,  # Currently this is hardcoded to OFF
        )
        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self,
        model: torch.nn.Module = None,
        accelerator: Accelerator = None,
        trainer: Trainer = None,
        pretrained_module_name_or_path: str = None,
    ):

        callbacks = []

        class ConvertAndSaveHFCheckpointAtEverySave(TrainerCallback):
            def __init__(self, pretrained_model_name_or_path: str, trainer: Trainer):
                self.pretrained_model_name_or_path = pretrained_model_name_or_path
                self.trainer = trainer

            def on_save(
                self,
                args: TrainingArguments,
                state: TrainerState,
                control: TrainerControl,
                **kwargs,
            ):
                """
                Save all HF files and convert dcp checkpoint to safetensors at every save operation.
                """

                async def checkpoint():
                    checkpoint_dir = os.path.join(
                        args.output_dir,
                        f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}",
                    )
                    hf_converted_output_dir = os.path.join(
                        checkpoint_dir, "hf_converted_checkpoint"
                    )
                    if os.path.exists(hf_converted_output_dir):
                        # if the folder already exists
                        # we return, since this is possible to happen
                        # saving the checkpointing at the end of the training
                        return
                    os.mkdir(hf_converted_output_dir)
                    try:
                        recover_safetensors_from_dcp(
                            checkpoint_dir,
                            self.pretrained_model_name_or_path,
                            hf_converted_output_dir,
                        )
                        # save tokenizer
                        if self.trainer.processing_class:
                            self.trainer.processing_class.save_pretrained(
                                hf_converted_output_dir
                            )
                        # save training args
                        torch.save(
                            args,
                            os.path.join(
                                hf_converted_output_dir, TRAINING_ARGS_NAME
                            ),
                        )
                        # save model config files
                        self.trainer.model.config.save_pretrained(
                            hf_converted_output_dir
                        )

                    except Exception as e:
                        raise ValueError(
                            f"Failed to convert the checkpoint {checkpoint_dir} to a HF compatible checkpoint"
                        ) from e
                if state.is_world_process_zero:
                    asyncio.run(checkpoint())

        callbacks.append(
            ConvertAndSaveHFCheckpointAtEverySave(
                pretrained_model_name_or_path=pretrained_module_name_or_path,
                trainer=trainer,
            )
        )
        if (
            accelerator is not None
            and getattr(accelerator.state, "fsdp_plugin", None) is not None
        ):

            # - use an internal function call to get the no split
            # module names, which are typically layers
            _layers = model._get_no_split_modules("")
            accelerator.state.fsdp_plugin.ignored_modules = [
                getattr(layer, name)
                for name in self._moe_component_module_names
                for layer in model.modules()
                if layer.__class__.__name__ in _layers
            ]

            # call this to patch the HF save and load functions to be able
            # to save DTensors propery
            patch_huggingface_save_and_load_for_dtensors()

            # call this to patch torch optim to not use
            # foreach for dtensors
            patch_torch_optim_foreach_to_not_apply_to_dtensors()

        return callbacks


# register
AccelerationPlugin.register_plugin(
    ScatterMoEAccelerationPlugin,
    configuration_and_paths=[
        "training.moe.scattermoe",
    ],
)
