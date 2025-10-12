# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Literal, Optional, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """
    blocks: tuple[list[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but 
    will be broken if we want to give different block_size to different 
    kv_cache_groups in the future.
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        return KVCacheBlocks(
            tuple(blk1 + blk2
                  for blk1, blk2 in zip(self.blocks, other.blocks)))

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]:
        ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> Optional[tuple[list[int], ...]]:
        ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> Optional[tuple[list[int], ...]]:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [
            block.block_id for block in self.blocks[0]
            if block.block_hash is None
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(tuple([] for _ in range(len(self.blocks))))


class KVCacheManager:

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        log_kv_cache_usage: bool = False,
        dcp_world_size: int = 1,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.log_kv_cache_usage = log_kv_cache_usage

        # log if vllm should log kv cache usage
        logger.info(f"KV cache usage logging is {'enabled' if log_kv_cache_usage else 'disabled'}")
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.block_size: Optional[int] = None
        if self.enable_caching:
            assert len(
                set(g.kv_cache_spec.block_size
                    for g in kv_cache_config.kv_cache_groups)
            ) == 1, "Only one block size is supported for now"
            self.block_size = kv_cache_config.kv_cache_groups[
                0].kv_cache_spec.block_size

            if dcp_world_size > 1:
                assert len(kv_cache_config.kv_cache_groups) == 1
                # Note(hc): need revisit. When both DCP and any future
                # PCP are enabled, the block_size may need to be scaled
                # by a factor of dcp_size × pcp_size?
                self.block_size *= dcp_world_size

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def _log_request_kv_cache_usage(self, request: Request, operation: str, 
                                   num_new_blocks: int = 0) -> None:
        """Log KV cache usage for a specific request.
        
        Args:
            request: The request to log KV cache usage for.
            operation: The operation being performed (e.g., 'allocate', 'free').
            num_new_blocks: Number of new blocks allocated in this operation.
        """
        if not self.log_kv_cache_usage:
            return
            
        try:
            request_id = request.request_id
            
            # Get blocks information for this request
            total_blocks_allocated = 0
            total_cached_blocks = 0
            
            for manager in self.coordinator.single_type_managers:
                if hasattr(manager, 'req_to_blocks') and request_id in manager.req_to_blocks:
                    req_blocks = manager.req_to_blocks[request_id]
                    total_blocks_allocated += len(req_blocks)
                    
                if hasattr(manager, 'num_cached_block') and request_id in manager.num_cached_block:
                    total_cached_blocks += manager.num_cached_block[request_id]
            
            # Calculate memory usage estimation
            block_size_tokens = self.block_size if self.block_size else 16
            memory_per_block_mb = self._estimate_block_memory_mb()
            total_memory_mb = total_blocks_allocated * memory_per_block_mb
            
            # Get global KV cache usage
            global_usage = self.usage
            free_blocks = self.block_pool.get_num_free_blocks()
            total_pool_blocks = self.block_pool.num_gpu_blocks
            
            logger.info(
                f"[KV-Cache-Usage] Request {request_id} - {operation}: "
                f"allocated_blocks={total_blocks_allocated}, "
                f"cached_blocks={total_cached_blocks}, "
                f"new_blocks={num_new_blocks}, "
                f"memory_est={total_memory_mb:.1f}MB, "
                f"tokens={request.num_tokens}, "
                f"block_size={block_size_tokens}, "
                f"global_usage={global_usage:.2%}, "
                f"free_blocks={free_blocks}/{total_pool_blocks}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to log KV cache usage for request {request_id}: {e}")

    def _estimate_block_memory_mb(self) -> float:
        """Estimate memory usage per block in MB.
        
        Returns:
            Estimated memory usage per block in MB.
        """
        try:
            if not self.kv_cache_config.kv_cache_groups:
                # No KV cache groups (e.g., attention-free models)
                logger.debug("No KV cache groups found, returning 0 memory usage")
                return 0.0
            
            # Calculate total memory per block across all KV cache groups
            total_memory_per_block_bytes = 0
            group_details = []
            
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
                kv_cache_spec = kv_cache_group.kv_cache_spec
                
                # Use the actual page_size_bytes from the KV cache spec
                # This accounts for all model parameters: block_size, num_kv_heads, 
                # head_size, dtype, and special cases like MLA
                page_size_bytes = kv_cache_spec.page_size_bytes
                
                # Number of layers in this group
                num_layers_in_group = len(kv_cache_group.layer_names)
                
                # Memory for this group
                group_memory_bytes = page_size_bytes * num_layers_in_group
                total_memory_per_block_bytes += group_memory_bytes
                
                # Store details for debugging
                group_details.append({
                    'group_id': i,
                    'layers': num_layers_in_group,
                    'page_size_bytes': page_size_bytes,
                    'group_memory_bytes': group_memory_bytes,
                    'spec_type': type(kv_cache_spec).__name__
                })
            
            # Convert bytes to MB
            memory_per_block_mb = total_memory_per_block_bytes / (1024 * 1024)
            
            # Log detailed breakdown if debug logging is enabled
            logger.debug(f"KV Cache memory estimation breakdown:")
            for detail in group_details:
                logger.debug(f"  Group {detail['group_id']} ({detail['spec_type']}): "
                        f"{detail['layers']} layers × {detail['page_size_bytes']} bytes = "
                        f"{detail['group_memory_bytes']} bytes")
            logger.debug(f"Total per block: {total_memory_per_block_bytes} bytes = "
                        f"{memory_per_block_mb:.2f} MB")
            
            return memory_per_block_mb
        except Exception:
            # Fallback estimation
            return 0.5  # 0.5 MB per block as rough estimate

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self,
                            request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # Prefix caching is disabled or
        # When the request requires prompt logprobs, we skip prefix caching.
        if (not self.enable_caching
                or (request.sampling_params is not None
                    and request.sampling_params.prompt_logprobs is not None)):
            return self.create_empty_block_list(), 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(request.block_hashes,
                                                    max_cache_hit_length))

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_new_computed_tokens

        return KVCacheBlocks(computed_blocks), num_new_computed_tokens

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed 
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = tuple(
                [] for _ in range(len(self.kv_cache_config.kv_cache_groups)))

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(request.request_id,
                                               request.num_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            self._log_request_kv_cache_usage(request, "allocate_failed", 0)
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not any(new_computed_block_list), (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.coordinator.save_new_computed_blocks(request.request_id,
                                                  new_computed_block_list)

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, num_tokens_need_slot, num_encoder_tokens)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_computed_tokens +
        # num_new_tokens, but must exclude "non-committable" tokens (e.g.,
        # draft tokens that could be rejected). Therefore, we cap the number
        # at `request.num_tokens`, ensuring only "finalized" tokens are cached.
        num_tokens_to_cache = min(num_computed_tokens + num_new_tokens,
                                  request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        # Log KV cache usage for this allocation
        total_new_blocks = sum(len(block_list) for block_list in new_blocks)
        self._log_request_kv_cache_usage(request, "allocate", total_new_blocks)

        return KVCacheBlocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        # Log KV cache usage before freeing
        self._log_request_kv_cache_usage(request, "free", 0)
        
        self.coordinator.free(request.request_id)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> list[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state for each kv cache group.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache 
            group.
        """
        assert request.status == RequestStatus.RUNNING
        return self.coordinator.get_num_common_prefix_blocks(
            request.request_id, num_running_requests)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return KVCacheBlocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled."""
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_empty_block_list(self) -> KVCacheBlocks:
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(tuple([]
                                   for _ in range(self.num_kv_cache_groups)))
