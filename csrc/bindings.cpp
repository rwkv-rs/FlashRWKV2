// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project
#include "bindings.h"
void register_flashrwkv2_bindings(py::module_& module) {
  register_recurrent_metadata_bindings(module);
  register_sampling_bindings(module);
}
