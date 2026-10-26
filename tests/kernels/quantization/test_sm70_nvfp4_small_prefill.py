# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-prefill stores preserve weights in both resident packed layouts."""

import io

import pytest
import torch


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize(
    "m,k,n", [(256, 128, 32), (512, 5120, 8704), (560, 4352, 5120), (1024, 128, 64)]
)
def test_small_prefill_matches_scalar_dequantized_basis(m, k, n, shared, gated):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    torch.manual_seed(123)
    q = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    packed = (q[:, ::2] | (q[:, 1::2] << 4)).contiguous()
    scale_bits = torch.randint(0, 127, (n, k // 16), device="cuda", dtype=torch.uint8)
    raw_scales = scale_bits.view(torch.float8_e4m3fn)
    codes, scales = torch.ops._C.nvfp4_qpn2_prepare_sm70(packed, raw_scales)
    # Representative checkpoint scale; the legacy converter materializes the
    # 16384-biased scale in FP16 before multiplying it by the FP4 code.
    global_scale = 0.0005
    reference_weight = torch.empty(k, n, device="cuda", dtype=torch.float16)
    torch.ops._C.nvfp4_qpn4_dequantize_sm70_out(
        reference_weight,
        codes.view(k, n // 2),
        scales.view(k // 16, n),
        global_scale,
        True,
    )
    columns = torch.arange(m, device="cuda") % k
    x = torch.zeros(m, k, device="cuda", dtype=torch.float16)
    x[torch.arange(m, device="cuda"), columns] = 1
    output = torch.empty(m, n // 2 if gated else n, device="cuda", dtype=x.dtype)
    if shared:
        effective = (raw_scales.float() * global_scale).half()
        tm_weight, tm_scales, meta = torch.ops._C.nvfp4_sm70_prepare(
            q.T.contiguous(), effective.T.contiguous(), 16, False
        )
        tm_scales.mul_(16384)
        torch.ops._C.nvfp4_qpn2_tm_dispatch_sm70_out(
            output,
            x,
            tm_weight,
            scales,
            global_scale,
            8,
            2,
            tm_scales,
            16,
            int(meta[0]),
            int(meta[1]),
            gated,
            256,
            True,
        )
    else:
        scratch = torch.empty_like(reference_weight)
        torch.ops._C.nvfp4_qpn4_prefill_sm70_out(
            output,
            scratch.data_ptr(),
            x,
            codes.view(k, n // 2),
            scales.view(k // 16, n),
            global_scale,
            True,
            gated,
        )
        # Exercise the real vector-store kernel, not a reimplementation.
        assert torch.equal(
            scratch.view(torch.uint8), reference_weight.view(torch.uint8)
        )
    expected = reference_weight.index_select(0, columns)
    if gated:
        gate, up = expected.chunk(2, dim=1)
        expected = (gate.float() / (1 + torch.exp(-gate.float()))).half() * up
        torch.testing.assert_close(output, expected, rtol=2e-3, atol=1e-3)
    else:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.isfinite().all()


@pytest.fixture
def qpn4_lazy_linear(request):
    """Prepare a real checkpoint-format layer without any workspace registry."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    import vllm._C  # noqa: F401

    from vllm.config.kernel import Sm70NvFp4Config
    from vllm.model_executor.kernels.linear.qpn.nvfp4 import (
        Qpn4NvFp4LinearKernel,
        Sm70NvFp4LinearLayerConfig,
    )

    gated = request.param
    # Regular dispatch uses split-K=17; gated dispatch uses split-K=8.
    k, n = 2176, 64
    torch.manual_seed(321)
    q = torch.randint(0, 16, (n, k), device="cuda", dtype=torch.uint8)
    packed = (q[:, ::2] | (q[:, 1::2] << 4)).contiguous()
    scales = (
        2.0 ** torch.randint(-2, 2, (n, k // 16), device="cuda").float()
    ).to(torch.float8_e4m3fn)
    global_scale = torch.tensor(0.125, device="cuda")
    # No prefix: registering a persistent layer workspace would raise.
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(packed, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(global_scale, requires_grad=False)
    policy = Sm70NvFp4Config(qpn4_lazy_dense_workspace=True)
    kernel = Qpn4NvFp4LinearKernel(
        Sm70NvFp4LinearLayerConfig(
            input_size=k,
            output_size=n,
            weight_shape=packed.shape,
            policy=policy,
            qpn2_qualified=False,
            qpn4_qualified=True,
            gated_silu=gated,
            dense_gated_silu=gated,
        )
    )
    assert kernel.try_process_weights(layer)
    # Independent NVFP4 reference; power-of-two scales avoid differences in
    # scale conversion rounding while exercising nonuniform per-block scales.
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    weight = (
        lut[q.long()] * scales.float().repeat_interleave(16, dim=1) * global_scale
    ).T.contiguous()

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            if gated:
                return kernel.apply_fused_silu_and_mul(self.layer, x)
            return kernel.apply_weights(self.layer, x)

    return Projection(), weight, gated


@pytest.mark.parametrize("qpn4_lazy_linear", [False, True], indirect=True)
@pytest.mark.parametrize("m", [1, 2, 17, 1025])
@pytest.mark.parametrize("execution", ["eager", "cudagraph", "export_reload"])
def test_qpn4_lazy_workspace_numeric_parity(qpn4_lazy_linear, m, execution):
    """Decode and both native prefill GEMMs retain real numerical behavior."""
    from vllm.model_executor.layers.quantization import sm70_turbomind as tm

    projection, weight, gated = qpn4_lazy_linear
    k, n = weight.shape
    x = torch.randn(m, k, device="cuda", dtype=torch.float16) / k**0.5
    if execution == "cudagraph":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                projection(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = projection(x)
        # Replay after changing the input: a captured M>1 temporary must remain
        # valid, while an M=1 graph must not require dense scratch at all.
        x.neg_()
        graph.replay()
    elif execution == "export_reload":
        artifact = io.BytesIO()
        torch.export.save(torch.export.export(projection, (x,)), artifact)
        artifact.seek(0)
        actual = torch.export.load(artifact).module()(x)
    else:
        actual = projection(x)
    state = getattr(projection.layer, tm.STATE_ATTR)
    scratch = torch.empty(k, n, device="cuda", dtype=torch.float16)
    resident_output = torch.empty(
        m, n // 2 if gated else n, device="cuda", dtype=torch.float16
    )
    # Compare against the unchanged resident-pointer contract as well as the
    # independent FP32 reference below. Neither path substitutes a fake kernel.
    state.native_ops.nvfp4_qpn4_dispatch_sm70_out(
        resident_output,
        scratch.data_ptr(),
        x,
        state.weight,
        state.scales,
        state.global_scale,
        state.use_scale_code,
        gated,
    )
    torch.testing.assert_close(actual, resident_output, rtol=0, atol=0)
    expected = x.float() @ weight
    if gated:
        gate, up = expected.chunk(2, dim=1)
        expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(actual.float(), expected, rtol=3e-3, atol=2e-3)
    assert actual.isfinite().all()
