from typing import List, Tuple

import cuda.bindings.driver as driver
import cuda.bindings.runtime as runtime
# import cuda.cudart as cudart
# import cuda.nvrtc as nvrtc
import torch
from cuda.bindings.driver import CUdevice, CUdevResource


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    # elif isinstance(error, runtime.cudaError_t):
    #     return cudart.cudaGetErrorName(error)[1]
    # elif isinstance(error, nvrtc.nvrtcResult):
    #     return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            f"CUDA error code={result[0].value}({_cudaGetErrorEnum(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]


def get_cudevice(dev: torch.device) -> CUdevice:
    try:
        # 初始化CUDA驱动程序
        try:
            checkCudaErrors(driver.cuInit(0))
        except RuntimeError:
            pass  # 如果已经初始化过，会出错，忽略即可

        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    except RuntimeError as e:
        runtime.cudaInitDevice(dev.index, 0, 0)
        cu_dev = checkCudaErrors(driver.cuDeviceGet(dev.index))
    return cu_dev


def get_device_resource(cu_dev: CUdevice) -> CUdevResource:
    return checkCudaErrors(
        driver.cuDeviceGetDevResource(
            cu_dev, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        )
    )


def split_resource(
    resource: CUdevResource,
    num_groups: int,
    min_count: int,
) -> Tuple[CUdevResource, CUdevResource]:
    results, _, remaining = checkCudaErrors(
        driver.cuDevSmResourceSplitByCount(
            num_groups,
            resource,
            0,  # useFlags
            min_count,
        )
    )
    return results, remaining


def create_green_ctx_streams(
    cu_dev: CUdevResource, resources: List[CUdevResource]
) -> List[torch.Stream]:
    streams = []
    for split in resources:
        desc = checkCudaErrors(driver.cuDevResourceGenerateDesc([split], 1))
        green_ctx = checkCudaErrors(
            driver.cuGreenCtxCreate(
                desc, cu_dev, driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
            )
        )
        stream = checkCudaErrors(
            driver.cuGreenCtxStreamCreate(
                green_ctx,
                driver.CUstream_flags.CU_STREAM_NON_BLOCKING,
                0,  # priority
            )
        )
        streams.append(torch.cuda.get_stream_from_external(stream))

    return streams


def split_device_green_ctx(
    dev: torch.device, num_groups: int, min_count: int
) -> Tuple[List[torch.Stream], List[CUdevResource]]:
    r"""
    Split the device into multiple `green contexts <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html>`_,
    return the corresponding streams and `CUdevResource` for each group and the remaining SMs.
    Green contexts allow concurrent execution of multiple kernels on different SM partitions.
    Args:
        dev: The device to split.
        num_groups: The number of groups to split the device into.
        min_count: Minimum number of SMs required for each group, it will be adjusted to meet the
            alignment and granularity requirements.
    Returns:
        streams: The list of torch.Streams objects corresponding to the green contexts.
        resources: The list of CUdevResource objects corresponding to the green contexts.
    Example:
        >>> from flashinfer.green_ctx import split_device_green_ctx
        >>> import torch
        >>> dev = torch.device("cuda:0")
        >>> streams, resources = split_device_green_ctx(dev, 2, 16)
        >>> print([r.sm.smCount for r in resources])
        [16, 16, 100]
        >>> with torch.cuda.stream(streams[0]):
        ...     x = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        ...     y = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
        ...     z = x @ y
        ...     print(z.shape)
        ...
        torch.Size([8192, 8192])
    Note:
        The length of the returned streams and resources is ``num_groups + 1``,
        where the last one is the remaining SMs.
    Raises:
        RuntimeError: when requested SM allocation exceeds device capacity:
        ``num_groups * round_up(min_count, 8) > num_sm``
    """
    cu_dev = get_cudevice(dev)
    resource = get_device_resource(cu_dev)
    results, remaining = split_resource(resource, num_groups, min_count)
    resources = results + [remaining]
    streams = create_green_ctx_streams(cu_dev, resources)
    return streams, resources


def test_different_batch_sizes(batch_sizes, output_path):
    dev = torch.device("cuda:0")
    num_group = 1

    stream0_sizes = range(2, 108, 2)

    f = open(output_path, "w")
    f.write("test,stream,Stream SM counts,elapsed time\n")

    for j, stream0_size in enumerate(stream0_sizes):
        try:
            streams, resources = split_device_green_ctx(
                dev, num_group, stream0_size)
            print(
                f"stream0 size={stream0_size}, SM counts: {[r.sm.smCount for r in resources]}"
            )

            num_streams = len(streams)

            # warm up
            for stream in streams:
                with torch.cuda.stream(stream):
                    x = torch.randn(8192, 8192, device=dev,
                                    dtype=torch.bfloat16)
                    y = torch.randn(8192, 8192, device=dev,
                                    dtype=torch.bfloat16)
                    z = x @ y

            start_events = [torch.cuda.Event(
                enable_timing=True) for _ in range(num_streams)]
            end_events = [torch.cuda.Event(enable_timing=True)
                          for _ in range(num_streams)]

            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    x = torch.randn(
                        batch_sizes[i], 8192, 8192, device=dev, dtype=torch.bfloat16)
                    y = torch.randn(8192, 8192, device=dev,
                                    dtype=torch.bfloat16)
                    start_events[i].record()
                    z = x @ y
                    end_events[i].record()
                    print("z shape:", z.shape)

            for stream in streams:
                stream.synchronize()

            for i in range(num_streams):
                elapsed_time = start_events[i].elapsed_time(end_events[i])
                print(
                    f"Stream {i}, Resource: {resources[i].sm.smCount}, Elapsed time: {elapsed_time} ms")

                f.write(f"{j},{i},{resources[i].sm.smCount},{elapsed_time}\n")

        except RuntimeError as e:
            print(f"stream0 size={stream0_size}, Error: {e}")

    # Test if we merge the batches into a single stream

    merged_batch_size = sum(batch_sizes)

    x = torch.randn(merged_batch_size, 8192, 8192,
                    device=dev, dtype=torch.bfloat16)
    y = torch.randn(8192, 8192, device=dev, dtype=torch.bfloat16)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    z = x @ y
    end_event.record()

    start_event.synchronize()
    end_event.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"Merged Stream, Elapsed time: {elapsed_time} ms")
    f.write(f"merged,0,{108},{elapsed_time}\n")


def test_parallel_streams(num_groups=2, min_count=16, matrix_size=8192):
    """
    测试不同 stream 内的任务并行地同时开启执行。
    
    Args:
        num_groups: 要创建的 green context 组数
        min_count: 每个组的最小 SM 数量
        matrix_size: 矩阵大小（用于矩阵乘法测试）
    """
    dev = torch.device("cuda:0")
    
    try:
        streams, resources = split_device_green_ctx(dev, num_groups, min_count)
        print(
            f"num_groups={num_groups}, min_count={min_count}, SM counts: {[r.sm.smCount for r in resources]}"
        )
        
        num_streams = len(streams)
        
        # Warm up
        for stream in streams:
            with torch.cuda.stream(stream):
                x = torch.randn(matrix_size, matrix_size, device=dev, dtype=torch.bfloat16)
                y = torch.randn(matrix_size, matrix_size, device=dev, dtype=torch.bfloat16)
                z = x @ y
        
        # 同步以确保 warm up 完成
        torch.cuda.synchronize()
        
        # 为每个 stream 创建事件
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_streams)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_streams)]
        
        # 记录总体开始时间
        overall_start = torch.cuda.Event(enable_timing=True)
        overall_start.record()
        
        # 在所有 streams 上同时启动任务（并行执行）
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                x = torch.randn(matrix_size, matrix_size, device=dev, dtype=torch.bfloat16)
                y = torch.randn(matrix_size, matrix_size, device=dev, dtype=torch.bfloat16)
                start_events[i].record(stream)
                z = x @ y
                end_events[i].record(stream)
        
        # 记录总体结束时间（在默认 stream 上）
        overall_end = torch.cuda.Event(enable_timing=True)
        overall_end.record()
        
        # 等待所有 streams 完成
        for stream in streams:
            stream.synchronize()
        
        # 同步以确保所有事件都已完成
        torch.cuda.synchronize()
        
        # 计算并打印每个 stream 的执行时间
        print("\n=== Parallel Stream Execution Results ===")
        for i in range(num_streams):
            elapsed_time = start_events[i].elapsed_time(end_events[i])
            print(
                f"Stream {i}, Resource: {resources[i].sm.smCount} SMs, "
                f"Elapsed time: {elapsed_time:.3f} ms"
            )
        
        # 计算总体执行时间
        overall_time = overall_start.elapsed_time(overall_end)
        print(f"\nOverall parallel execution time: {overall_time:.3f} ms")
        print(f"Number of streams: {num_streams}")
        
    except RuntimeError as e:
        print(f"num_groups={num_groups}, min_count={min_count}, Error: {e}")


def test_simple():
    dev = torch.device("cuda:0")

    num_groups_list = [1, 2, 3]
    min_count_list = [16, 32]

    for num_groups in num_groups_list:
        for min_count in min_count_list:
            try:
                streams, resources = split_device_green_ctx(
                    dev, num_groups, min_count)
                print(
                    f"num_groups={num_groups}, min_count={min_count}, SM counts: {[r.sm.smCount for r in resources]}"
                )

                for i, stream in enumerate(streams):
                    with torch.cuda.stream(stream):

                        start_time = torch.cuda.Event(enable_timing=True)
                        start_time.record()
                        x = torch.randn(8192, 8192, device=dev,
                                        dtype=torch.bfloat16)
                        y = torch.randn(8192, 8192, device=dev,
                                        dtype=torch.bfloat16)

                        allocate_time = torch.cuda.Event(enable_timing=True)
                        allocate_time.record()
                        torch.cuda.synchronize()
                        elapsed_time = start_time.elapsed_time(allocate_time)
                        print(
                            f"Stream {i}, Resource: {resources[i].sm.smCount}, Allocation time: {elapsed_time} ms")
                        start_time.record()
                        z = x @ y
                        end_time = torch.cuda.Event(enable_timing=True)
                        end_time.record()
                        torch.cuda.synchronize()
                        elapsed_time = start_time.elapsed_time(end_time)
                        print(
                            f"Stream {i}, Resource: {resources[i].sm.smCount}, Elapsed time: {elapsed_time} ms")

                        print(stream, z.shape)
            except RuntimeError as e:
                print(
                    f"num_groups={num_groups}, min_count={min_count}, Error: {e}"
                )


if __name__ == "__main__":
    # Example usage
    import sys

    output_path = sys.argv[1] if len(
        sys.argv) > 1 else "green_ctx_batch_size_test.csv"

    batch_sizes = [1, 32]

    # test_simple()
    # test_different_batch_sizes(batch_sizes, output_path)


    test_parallel_streams(1, 16, 8192)
    test_parallel_streams(2, 16, 8192)
    test_parallel_streams(3, 16, 8192)
    