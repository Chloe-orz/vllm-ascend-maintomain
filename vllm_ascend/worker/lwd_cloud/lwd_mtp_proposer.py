# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Feed the MTP first pass with the same remote prompt rows as the target."""

import torch
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_ascend.spec_decode.eagle_proposer import AscendEagleProposer


class LwdMTPProposer(AscendEagleProposer):
    def _backup_token_id(self, req_state: CachedRequestState, idx: int) -> int:
        # Cloud prompt IDs are unknown. Zero only fills the backup slot for
        # unfinished prefill; the first-pass hook supplies remote prompt embeds.
        if req_state.prompt_token_ids is None and 0 <= idx < req_state.num_prompt_tokens:
            return 0
        return req_state.get_token_id(idx)

    def _prepare_first_pass_inputs_embeds(
        self,
        inputs_embeds: torch.Tensor | None,
        num_tokens: int,
        num_input_tokens: int,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> torch.Tensor | None:
        runner = self.runner
        if runner is None:
            raise RuntimeError("[LWD] MTP requires its target model runner")
        computed = runner.input_batch.num_computed_tokens_cpu
        prompt_lengths = runner.input_batch.num_prompt_tokens
        needs_prompt = [int(computed[i]) + 1 < int(prompt_lengths[i]) for i in range(runner.input_batch.num_reqs)]
        if not any(needs_prompt):
            return inputs_embeds
        # Native first-pass preparation already shifted input_ids by one row
        # and inserted the sampled next token at each request's final row.
        out = self.inputs_embeds
        out[:num_tokens] = self.model.embed_input_ids(self.input_ids[:num_tokens])
        starts = common_attn_metadata.query_start_loc_cpu.tolist()
        for i, has_prompt in enumerate(needs_prompt):
            if has_prompt:
                start, end = starts[i : i + 2]
                out[start : end - 1] = runner.inputs_embeds.gpu[start + 1 : end]
        if num_input_tokens > num_tokens:
            out[num_tokens:num_input_tokens].zero_()
        # The current models perform their own SP slicing for inputs_embeds.
        return out[:num_input_tokens]
