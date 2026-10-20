// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
#pragma once
#include <torch/extension.h>
void register_flashrwkv2_bindings(py::module_&);
void register_recurrent_metadata_bindings(py::module_&);
void register_sampling_bindings(py::module_&);
