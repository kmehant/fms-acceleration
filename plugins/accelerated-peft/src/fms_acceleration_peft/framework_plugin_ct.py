# Copyright The IBM Tuning Team
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

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Standard
from typing import Dict, Tuple
import inspect
import os
import warnings

# Third Party
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig, get_peft_model
from peft.utils.other import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers.utils.import_utils import _is_package_available
import torch

# Local
from .fsdp_utils import put_selected_meta_tensors_on_cpu
from .compressed_tensors_utils import (
    register_tensors_as_parameters_patch_rule
)

def _prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None
):

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for _, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        ):
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(_module, _input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        # To support older transformers versions,
        # check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers.",
                "The passed kwargs will be ignored. if you want to use that feature,",
                "please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {}
            if not _supports_gc_kwargs
            else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model


def register_prehook(target_module, torch_dtype, model):
    def prehook_recover_original_dtype(module, input):
        # print(input)
        # print(input[0].size())
        if hasattr(module, 'weight') and module.weight is not None:
            attr = getattr(module, "weight")
            attr = torch.nn.Parameter(attr.to(torch.int8), requires_grad=False)
            setattr(module, "weight", attr)
            module.weight.data = module.weight.data.to(torch.int32)
    for m in model.modules():
        if isinstance(m, target_module):
            # attr = getattr(m, "weight")
            # attr = torch.nn.Parameter(attr.to(torch_dtype), requires_grad=False)
            # setattr(m, "weight", attr)
            # attr = getattr(m, "weight_scale")
            # attr = torch.nn.Parameter(attr.view(torch_dtype), requires_grad=False)
            # setattr(m, "weight_scale", attr)
            m.weight.data = m.weight.to(torch_dtype)
        # m.register_forward_pre_hook(prehook_recover_original_dtype)


class CompressedTensorsAccelerationPlugin(AccelerationPlugin):

    require_packages = ["compressed_tensors"]

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)
        self._no_peft_model = self._check_config_and_maybe_check_values(
            key="peft.quantization.compressed_tensors.no_peft_model", values=[True, False]
        )
        print(self._no_peft_model)


    def model_loader(self, model_name: str, **kwargs):

        # get additional parameters
        torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        low_cpu_mem_usage = kwargs.get("low_cpu_mem_usage")
        attn_implementation = kwargs.get("attn_implementation")

        # config_kwargs = {}
        try:
            world_size = torch.distributed.get_world_size()
        except ValueError:
            world_size = 1  # pg not init

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            token=None,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        )

        if (
            world_size > 1
            and os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        ):

            # register FSDP patch
            # results in mat1 and mat2 shapes cannot be multiplied
            from compressed_tensors.linear.compressed_linear import CompressedLinear
            register_tensors_as_parameters_patch_rule(
                target_module=CompressedLinear,
                torch_dtype=torch_dtype,
            )
            # register_prehook(target_module=CompressedLinear, torch_dtype=torch_dtype, model=model)

            # _, _transformers_version = _is_package_available(
            #     "transformers", return_version=True
            # )
            # _trl_installed, _trl_version = _is_package_available(
            #     "trl", return_version=True
            # )

            # if _transformers_version >= "4.45" and (
            #     not _trl_installed or (_trl_installed and _trl_version >= "0.12")
            # ):
            #     # in low_cpu_mem_mode, if certain tensors like embeddings
            #     # are in the meta device, then certain operations like
            #     # embedding resizing will fail
            #     put_selected_meta_tensors_on_cpu(model)
        return model

    @property
    def requires_custom_loading(self):
        return True

    @property
    def requires_augmentation(self):
        # will skip the augmentation if _no_peft_model == True
        print(self._no_peft_model)
        return not self._no_peft_model

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        # - when using our prepare peft, we will enforce the mixed precision settings
        # assert (
        #     train_args.bf16 is True or train_args.fp16 is True
        # ), f"{self.__class__} requires mixed precision argument `--fp16` or `--bf16`"

        (peft_config,) = modifiable_args  # unpack modifiable args

        # some assertions
        assert peft_config is not None, "need peft_config to install PEFT adapters"

        # requires a custom prepare because the stock one in peft will introduce
        # extraneous casting
        model = _prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=train_args.gradient_checkpointing_kwargs,
        )

        model = get_peft_model(model, peft_config)
        for n, p  in model.named_parameters():
            if p.requires_grad:
                print(n, p.dtype)
        modifiable_args = (None,)  # return a None
        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):
        callbacks = []
        if (
            accelerator is not None
            and getattr(accelerator.state, "fsdp_plugin", None) is not None
        ):
            _, _transformers_version = _is_package_available(
                "transformers", return_version=True
            )
            _trl_installed, _trl_version = _is_package_available(
                "trl", return_version=True
            )

            # the meta device fix for quantized models is since this transformers version
            # or if trl is installed then its only for this version
            if _transformers_version >= "4.45" and (
                not _trl_installed or (_trl_installed and _trl_version >= "0.11.4")
            ):
                # guarded
                # NOTE: replace this later with a more specific accelerate version check
                try:
                    # Third Party
                    # pylint: disable=import-outside-toplevel
                    from torch.distributed.utils import ensure_weights_retied

                    # then its handled internally and there is nothing to do
                except ImportError:
                    # need to use our internal version
                    # Local
                    from .fsdp_utils import (  # pylint: disable=import-outside-toplevel
                        ensure_weights_retied,
                    )

                    accelerator.state.fsdp_plugin.param_init_fn = ensure_weights_retied(
                        accelerator.state.fsdp_plugin.param_init_fn,
                        model if self._no_peft_model else model.get_base_model(),
                        accelerator.device,
                    )

        return callbacks


# register
AccelerationPlugin.register_plugin(
    CompressedTensorsAccelerationPlugin,
    configuration_and_paths=["peft.quantization.compressed_tensors"],
)
