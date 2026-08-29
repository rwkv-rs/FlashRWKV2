// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the FlashRWKV2 project

#include <Python.h>

#if defined(FLASHRWKV_BACKEND_SM90)
#define FLASHRWKV_MODULE_NAME _C_sm90
#elif defined(FLASHRWKV_BACKEND_SM120)
#define FLASHRWKV_MODULE_NAME _C_sm120
#else
#error "FlashRWKV2 private extension requires an architecture backend macro"
#endif

#define FLASHRWKV_CONCAT_IMPL(left, right) left##right
#define FLASHRWKV_CONCAT(left, right) FLASHRWKV_CONCAT_IMPL(left, right)
#define FLASHRWKV_STRINGIFY_IMPL(value) #value
#define FLASHRWKV_STRINGIFY(value) FLASHRWKV_STRINGIFY_IMPL(value)

namespace {

PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    nullptr,
    nullptr,
    -1,
    nullptr,
};

}  // namespace

PyMODINIT_FUNC FLASHRWKV_CONCAT(PyInit_, FLASHRWKV_MODULE_NAME)() {
  module_definition.m_name =
      "flashrwkv2." FLASHRWKV_STRINGIFY(FLASHRWKV_MODULE_NAME);
  return PyModule_Create(&module_definition);
}
