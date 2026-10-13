# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Private P2P symbols on a torch_npu-owned HCCL communicator."""

import ctypes

import torch

from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    aclrtStream_t,
    buffer_type,
    hcclComm_t,
    hcclDataType_t,
    hcclDataTypeEnum,
    hcclResult_t,
)
from vllm_ascend.utils import find_hccl_library


class LwdHcclLibrary:
    def __init__(self) -> None:
        self.library_path = find_hccl_library()
        self._lib = ctypes.CDLL(self.library_path)
        self._error = self._lib.HcclGetErrorString
        self._error.argtypes = [hcclResult_t]
        self._error.restype = ctypes.c_char_p
        self._functions = {}
        for name in ("HcclSend", "HcclRecv"):
            function = getattr(self._lib, name)
            function.argtypes = [
                buffer_type,
                ctypes.c_uint64,
                hcclDataType_t,
                ctypes.c_uint32,
                hcclComm_t,
                aclrtStream_t,
            ]
            function.restype = hcclResult_t
            self._functions[name] = function

    def _p2p(self, name: str, tensor: torch.Tensor, peer: int, comm: hcclComm_t) -> None:
        result = self._functions[name](
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            hcclDataTypeEnum.from_torch(tensor.dtype),
            peer,
            comm,
            aclrtStream_t(torch.npu.current_stream().npu_stream),
        )
        if result:
            detail = self._error(result)
            raise RuntimeError(f"[LWD] {name} failed: {detail!r} (code={result})")

    def send(self, tensor: torch.Tensor, dest_rank: int, comm: hcclComm_t) -> None:
        self._p2p("HcclSend", tensor, dest_rank, comm)

    def recv(self, tensor: torch.Tensor, src_rank: int, comm: hcclComm_t) -> None:
        self._p2p("HcclRecv", tensor, src_rank, comm)
