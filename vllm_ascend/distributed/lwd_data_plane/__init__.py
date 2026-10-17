# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""prefill_only LWD data plane: duplex comm package.

Two physical channels, direction-only (no per-type channels):

    UP   edge -> cloud : whole-prompt embeddings, one message per request
    DOWN cloud -> edge : combined c2e packet (final hidden + topk +
                         num_accepted of the last decode step), one
                         message per request

The machinery (send snapshot, per-channel seqno reorder buffer, lazy
reap, HCCL->stream bridge) is ported from the demo branch's
``lwd_comm`` package with draft/plain wire variants removed —
the payloads here are always single contiguous bf16 tensors whose shape
the receiver learns from the control plane before posting the irecv.
"""
