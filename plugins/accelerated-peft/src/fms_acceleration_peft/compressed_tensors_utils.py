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
from functools import partial
from typing import Callable, List

# Third Party
from fms_acceleration.model_patcher import (
    ModelPatcher,
    ModelPatcherRule,
    ModelPatcherTrigger,
)
import torch

# these parameters are to be patched for triton v2
# consider making a map if patching more kernels
PATCH_FOR_FSDP_COMPRESSED_TENSORS = ["weight"]



def build_patch_to_view_tensor_to_parameter_for_fsdp_compressed_tensors(
    module,
    torch_dtype,
):
    # convert all patched attributes to Parameters of torch_dtype
    # so FSDP can shard them
    for attr_name in PATCH_FOR_FSDP_COMPRESSED_TENSORS:
        attr = getattr(module, attr_name)
        attr = torch.nn.Parameter(attr.to(torch_dtype), requires_grad=False)
        setattr(module, attr_name, attr)

    # this patches the forward to convert them back to original
    # type (i.e. int32) before the function call into the kernels
    # return module.forward
    return patch_forward_to_view_attributes_before_call(
        module.forward,
        attribute_names=PATCH_FOR_FSDP_COMPRESSED_TENSORS,
        torch_dtype=torch.int8,
    )


def register_tensors_as_parameters_patch_rule(target_module, torch_dtype):
    # Register patch
    ModelPatcher.register(
        ModelPatcherRule(
            rule_id="compressed_tensors_patch_tensors_as_float_parameters",
            trigger=ModelPatcherTrigger(check=target_module),
            forward_builder=partial(
                build_patch_to_view_tensor_to_parameter_for_fsdp_compressed_tensors,
                torch_dtype=torch_dtype,
            ),
        )
    )




# consider to move this somewhere more general
def patch_forward_to_view_attributes_before_call(
    old_forward: Callable,
    attribute_names: List[str],
    torch_dtype: torch.dtype,
    submodule_names: str = None,
    is_method_forward: bool = True,
):
    # patch old_forward to view attribtues to torch_dype
    # before call

    if submodule_names is None:
        submodule_names = ""
    if isinstance(submodule_names, str):
        submodule_names = [submodule_names]

    def _forward(self, *args, **kwargs):

        for sub_name in submodule_names:
            mod = self.get_submodule(sub_name)

            # perform a view on all these attributes
            for attr_name in attribute_names:

                # the view should be a passthrough
                # if attr.dtype == torch_dtype
                attr = getattr(mod, attr_name)

                # perform view
                attr = attr.view(torch_dtype)

                try:
                    setattr(mod, attr_name, attr)
                except TypeError:
                    # this means already have attr_name as a parameter, then
                    # just assign this way
                    mod.__dict__[attr_name] = attr

        if is_method_forward:
            # in this case, the self is already bound
            return old_forward(*args, **kwargs)
        return old_forward(self, *args, **kwargs)

    return _forward
