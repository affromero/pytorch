import asyncio
import io
import os
import pickle
import pickletools
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


class TrustedState:
    def __init__(self, value):
        self.value = value


class OtherState:
    pass


def clear_permissions():
    torch.serialization.clear_safe_globals()
    return None


class ClearScope:
    def __reduce__(self):
        return clear_permissions, ()


class ListAssignment:
    def __reduce__(self):
        return list, ([0],), None, None, iter([(0, 17)])


class ExecutableState:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({self.marker!r}).write_text('executed')",)


def checkpoint(value, **kwargs):
    buffer = io.BytesIO()
    torch.save(value, buffer, **kwargs)
    buffer.seek(0)
    return buffer


class RestrictedLoadingTests(unittest.TestCase):
    def tearDown(self):
        torch.serialization.clear_safe_globals()

    def test_restricted_loading_rejects_torchscript_and_explicit_jit_loading_preserves_inference(self):
        expected = torch.arange(4, dtype=torch.float32)
        module = torch.jit.trace(torch.nn.ReLU(), expected)
        buffer = io.BytesIO()
        torch.jit.save(module, buffer)
        buffer.seek(0)
        with self.assertRaisesRegex(RuntimeError, "TorchScript archive"):
            torch.load(buffer, weights_only=True)
        buffer.seek(0)
        loaded = torch.jit.load(buffer, map_location="cpu")
        torch.testing.assert_close(loaded(expected), module(expected))

    def test_legacy_tar_is_rejected_before_unpickling(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for name in ("storages", "tensors", "pickle"):
                data = pickle.dumps(0, protocol=2)
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        buffer.seek(0)
        with tempfile.NamedTemporaryFile() as handle:
            handle.write(buffer.getvalue())
            handle.flush()
            with self.assertRaisesRegex((RuntimeError, pickle.UnpicklingError), "legacy .tar"):
                torch.load(handle.name, weights_only=True)

    def test_old_non_tar_tensor_checkpoints_remain_supported(self):
        expected = {"weights": torch.arange(8), "label": "model"}
        result = torch.load(checkpoint(expected, _use_new_zipfile_serialization=False), weights_only=True)
        torch.testing.assert_close(result["weights"], expected["weights"])
        self.assertEqual(result["label"], expected["label"])

    def test_untrusted_class_is_rejected_until_explicitly_allowed(self):
        buffer = checkpoint(TrustedState(17))
        with self.assertRaises((RuntimeError, pickle.UnpicklingError)):
            torch.load(buffer, weights_only=True)
        buffer.seek(0)
        with torch.serialization.safe_globals([TrustedState]):
            self.assertEqual(torch.load(buffer, weights_only=True).value, 17)
        self.assertEqual(torch.serialization.get_safe_globals(), [])

    def test_nested_overlapping_scopes_restore_permissions_after_exception(self):
        torch.serialization.add_safe_globals([TrustedState])
        with self.assertRaisesRegex(ValueError, "scope failure"):
            with torch.serialization.safe_globals([TrustedState, OtherState]):
                with torch.serialization.safe_globals([TrustedState]):
                    self.assertTrue(any(item is OtherState for item in torch.serialization.get_safe_globals()))
                self.assertTrue(any(item is OtherState for item in torch.serialization.get_safe_globals()))
                raise ValueError("scope failure")
        self.assertEqual(torch.serialization.get_safe_globals(), [TrustedState])

    def test_concurrent_loader_cannot_borrow_another_threads_permissions(self):
        data = checkpoint(TrustedState(23)).getvalue()
        barrier = threading.Barrier(2)

        def allowed():
            with torch.serialization.safe_globals([TrustedState]):
                barrier.wait(timeout=10)
                value = torch.load(io.BytesIO(data), weights_only=True)
                barrier.wait(timeout=10)
                return value.value

        def unrelated():
            barrier.wait(timeout=10)
            try:
                with self.assertRaises((RuntimeError, pickle.UnpicklingError)):
                    torch.load(io.BytesIO(data), weights_only=True)
            finally:
                barrier.wait(timeout=10)

        with ThreadPoolExecutor(max_workers=2) as pool:
            trusted = pool.submit(allowed)
            untrusted = pool.submit(unrelated)
            self.assertEqual(trusted.result(timeout=20), 23)
            untrusted.result(timeout=20)

    def test_async_scope_does_not_grant_permissions_to_an_unrelated_task(self):
        data = checkpoint(TrustedState(31)).getvalue()

        async def exercise():
            ready = asyncio.Event()
            finished = asyncio.Event()

            async def allowed():
                with torch.serialization.safe_globals([TrustedState]):
                    ready.set()
                    await finished.wait()
                    self.assertEqual(torch.load(io.BytesIO(data), weights_only=True).value, 31)

            async def unrelated():
                await ready.wait()
                try:
                    with self.assertRaises((RuntimeError, pickle.UnpicklingError)):
                        torch.load(io.BytesIO(data), weights_only=True)
                finally:
                    finished.set()

            await asyncio.gather(allowed(), unrelated())

        asyncio.run(exercise())

    def test_numpy_rng_checkpoint_restores_the_exact_random_continuation(self):
        rng = np.random.RandomState(1729)
        rng.random_sample(13)
        buffer = checkpoint({"numpy": rng.get_state(), "cpu": torch.get_rng_state()})
        expected = rng.random_sample(50)
        with torch.serialization.safe_globals([
            np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))
        ]):
            restored = torch.load(buffer, weights_only=True)
        rng.set_state(restored["numpy"])
        np.testing.assert_array_equal(rng.random_sample(50), expected)
        torch.testing.assert_close(restored["cpu"], torch.get_rng_state())

    def test_blocked_operating_system_function_cannot_be_allowlisted(self):
        with torch.serialization.safe_globals([os.system]):
            with self.assertRaises((RuntimeError, pickle.UnpicklingError)):
                torch.load(checkpoint(os.system), weights_only=True)

    def test_loader_keeps_its_permission_snapshot_when_context_changes_during_load(self):
        with torch.serialization.safe_globals([clear_permissions, TrustedState]):
            result = torch.load(checkpoint([ClearScope(), TrustedState(43)]), weights_only=True)
            self.assertEqual(result[1].value, 43)
            self.assertEqual(torch.serialization.get_safe_globals(), [])

    def test_pickle_dictionary_assignment_cannot_modify_a_list(self):
        with torch.serialization.safe_globals([list]):
            with self.assertRaises((RuntimeError, pickle.UnpicklingError)):
                torch.load(checkpoint(ListAssignment()), weights_only=True)


class QuantizedStateTests(unittest.TestCase):
    def test_state_round_trip_preserves_exact_qparams_and_inference(self):
        for nested in (False, True):
            with self.subTest(nested=nested):
                source = torch.ao.nn.quantized.Sigmoid(0.03, 7)
                target = torch.ao.nn.quantized.Sigmoid(0.125, 0)
                if nested:
                    source, target = torch.nn.Sequential(source), torch.nn.Sequential(target)
                target.load_state_dict(torch.load(checkpoint(source.state_dict()), weights_only=True))
                restored = target[0] if nested else target
                self.assertEqual(restored.output_scale, 0.03)
                self.assertEqual(restored.output_zero_point, 7)
                inputs = torch.quantize_per_tensor(torch.tensor([-2.0, 0.0, 2.0]), 0.1, 20, torch.quint8)
                expected, actual = source(inputs), target(inputs)
                self.assertEqual(actual.q_scale(), expected.q_scale())
                self.assertEqual(actual.q_zero_point(), expected.q_zero_point())
                torch.testing.assert_close(actual.int_repr(), expected.int_repr())
                torch.testing.assert_close(actual.dequantize(), expected.dequantize())

    def test_missing_and_unexpected_state_keys_remain_visible(self):
        target = torch.ao.nn.quantized.Sigmoid(0.03, 7)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            target.load_state_dict({})
        result = target.load_state_dict({}, strict=False)
        self.assertEqual(set(result.missing_keys), {"output_scale", "output_zero_point"})
        state = target.state_dict()
        state["unrecognized"] = torch.tensor(1)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            target.load_state_dict(state)

    def test_malformed_qparams_do_not_partially_change_inference_state(self):
        invalid = [
            ("output_scale", torch.tensor([0.03])),
            ("output_scale", torch.tensor(float("nan"))),
            ("output_scale", torch.tensor(float("inf"))),
            ("output_scale", torch.tensor(0.0)),
            ("output_scale", torch.tensor(-0.1)),
            ("output_zero_point", torch.tensor(1.5)),
            ("output_zero_point", torch.tensor([1])),
        ]
        for name, value in invalid:
            with self.subTest(name=name, value=value):
                target = torch.ao.nn.quantized.Sigmoid(0.03, 7)
                state = target.state_dict()
                state[name] = value
                with self.assertRaisesRegex(RuntimeError, name):
                    target.load_state_dict(state)
                self.assertEqual((target.output_scale, target.output_zero_point), (0.03, 7))

    def test_scripted_module_save_load_preserves_inference(self):
        source = torch.jit.script(torch.ao.nn.quantized.Sigmoid(0.03, 7))
        buffer = io.BytesIO()
        torch.jit.save(source, buffer)
        buffer.seek(0)
        restored = torch.jit.load(buffer)
        inputs = torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0]), 0.1, 20, torch.quint8)
        torch.testing.assert_close(restored(inputs).int_repr(), source(inputs).int_repr())
        self.assertEqual(restored(inputs).q_scale(), source(inputs).q_scale())
        self.assertEqual(restored(inputs).q_zero_point(), source(inputs).q_zero_point())


class NativeInputValidationTests(unittest.TestCase):
    def run_isolated(self, source):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", "import torch\n" + source],
                cwd=directory, text=True, capture_output=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_empty_packed_sequence_is_rejected_without_crashing(self):
        self.run_isolated('''
try:
    torch.ops.aten._pad_packed_sequence(torch.empty(0), torch.empty(0, dtype=torch.int64), False, 0.0, 0)
except RuntimeError as error:
    assert "empty" in str(error)
else:
    raise AssertionError("empty batch sizes accepted")
''')

    def test_tensor_storage_size_must_match_the_archive_record(self):
        data = checkpoint(torch.arange(4, dtype=torch.float32)).getvalue()
        for size_change in (-1, 1):
            with self.subTest(size_change=size_change):
                modified = io.BytesIO()
                with zipfile.ZipFile(io.BytesIO(data)) as source:
                    with zipfile.ZipFile(modified, "w") as destination:
                        for name in source.namelist():
                            value = source.read(name)
                            if name.endswith("/data/0"):
                                value = value[:-1] if size_change < 0 else value + b"\x00"
                            destination.writestr(name, value)
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "checkpoint.pt")
                    with open(path, "wb") as handle:
                        handle.write(modified.getvalue())
                    for mmap in (False, True):
                        with self.subTest(mmap=mmap):
                            with self.assertRaisesRegex(pickle.UnpicklingError, "record size"):
                                torch.load(path, weights_only=True, mmap=mmap)

    def test_shared_tensor_storage_round_trip_preserves_aliasing_with_mmap(self):
        value = torch.arange(12, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            torch.save({"base": value, "view": value[3:7]}, path)
            for mmap in (False, True):
                with self.subTest(mmap=mmap):
                    loaded = torch.load(path, weights_only=True, mmap=mmap)
                    torch.testing.assert_close(loaded["base"], value)
                    loaded["view"][0] = 43
                    self.assertEqual(loaded["base"][3].item(), 43)

    def test_invalid_storage_counts_are_rejected_before_materialization(self):
        data = checkpoint(torch.arange(4, dtype=torch.float32)).getvalue()
        for replacement in (b"J\xff\xff\xff\xff", b"\x88"):
            with self.subTest(replacement=replacement):
                modified = io.BytesIO()
                with zipfile.ZipFile(io.BytesIO(data)) as source, zipfile.ZipFile(modified, "w") as target:
                    for name in source.namelist():
                        value = source.read(name)
                        if name.endswith("/data.pkl"):
                            operations = list(pickletools.genops(value))
                            persistent = next(i for i, (op, _, _) in enumerate(operations) if op.name == "BINPERSID")
                            count = next(i for i in range(persistent - 1, -1, -1)
                                         if operations[i][0].name == "BININT1")
                            start, end = operations[count][2], operations[count + 1][2]
                            value = value[:start] + replacement + value[end:]
                        target.writestr(name, value)
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "checkpoint.pt")
                    with open(path, "wb") as handle:
                        handle.write(modified.getvalue())
                    for mmap in (False, True):
                        with self.assertRaisesRegex(pickle.UnpicklingError, "element count"):
                            torch.load(path, weights_only=True, mmap=mmap)

    def test_rnn_cells_reject_inconsistent_gate_weights(self):
        for name, gates in (("lstm_cell", 4), ("gru_cell", 3), ("rnn_tanh_cell", 1), ("rnn_relu_cell", 1)):
            with self.subTest(cell=name):
                self.run_isolated(f'''
inputs, hidden = torch.zeros(1, 2), torch.zeros(1, 2)
hidden_arg = [hidden, hidden] if {name!r} == "lstm_cell" else hidden
try:
    getattr(torch.ops.aten, {name!r})(inputs, hidden_arg, torch.zeros({gates} * 2 + 1, 2), torch.zeros({gates} * 2, 2))
except RuntimeError as error:
    assert "weight_ih" in str(error)
else:
    raise AssertionError("inconsistent weights accepted")
''')

    def test_rnn_cells_reject_inconsistent_hidden_gate_weights(self):
        for name, gates in (("lstm_cell", 4), ("gru_cell", 3), ("rnn_tanh_cell", 1), ("rnn_relu_cell", 1)):
            with self.subTest(cell=name):
                self.run_isolated(f'''
inputs, hidden = torch.zeros(1, 2), torch.zeros(1, 2)
hidden_arg = [hidden, hidden] if {name!r} == "lstm_cell" else hidden
try:
    getattr(torch.ops.aten, {name!r})(inputs, hidden_arg, torch.zeros({gates} * 2, 2), torch.zeros({gates} * 2 + 1, 2))
except RuntimeError as error:
    assert "weight_hh" in str(error)
else:
    raise AssertionError("inconsistent hidden weights accepted")
''')

    def test_empty_ctc_input_is_rejected_without_crashing(self):
        self.run_isolated('''
try:
    torch.nn.functional.ctc_loss(torch.empty(0, 0, 4), torch.empty(0, dtype=torch.int64), [], [], reduction="none")
except RuntimeError as error:
    assert "empty" in str(error)
else:
    raise AssertionError("empty CTC input accepted")
''')

    def test_bare_container_annotations_fail_cleanly(self):
        for name in ("list", "tuple"):
            with self.subTest(container=name):
                self.run_isolated(f'''
try:
    torch.jit.CompilationUnit("def invalid():\\n    x: {name} = []\\n    return x\\n")
except RuntimeError as error:
    assert "without a contained type" in str(error)
else:
    raise AssertionError("bare container annotation accepted")
''')

    def test_profiler_rejects_invalid_handles_synchronously(self):
        for dtype in ("torch.int64", "torch.uint8"):
            for completed in (False, True):
                with self.subTest(dtype=dtype, completed=completed):
                    self.run_isolated(f'''
future = torch.futures.Future()
if {completed!r}:
    future.set_result(17)
try:
    torch.ops.profiler._call_end_callbacks_on_jit_fut(torch.tensor(0, dtype={dtype}), future)
except RuntimeError:
    pass
else:
    raise AssertionError("invalid handle accepted")
''')

    def test_profiler_null_future_does_not_crash(self):
        self.run_isolated('''
try:
    torch.ops.profiler._call_end_callbacks_on_jit_fut(torch.tensor(0), None)
except RuntimeError:
    pass
else:
    raise AssertionError("null future accepted")
''')

    def test_profiler_valid_records_reject_null_futures(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                self.run_isolated(f'''
record = torch.ops.profiler.{'_record_function_enter' if legacy else '_record_function_enter_new'}("security-regression", None)
try:
    torch.ops.profiler._call_end_callbacks_on_jit_fut{'' if legacy else '._RecordFunction'}(record, None)
except RuntimeError as error:
    assert "Future" in str(error) or "future" in str(error)
else:
    raise AssertionError("null future accepted")
''')

    @unittest.skipUnless(torch.backends.mkldnn.is_available(), "MKLDNN is unavailable on this build")
    def test_mkldnn_pooling_rejects_nonpositive_stride_without_crashing(self):
        for stride in (0, -1):
            with self.subTest(stride=stride):
                self.run_isolated(f'''
assert torch.backends.mkldnn.is_available(), "MKLDNN build is required for this regression"
try:
    torch.nn.functional.max_pool2d(torch.ones(1, 1, 4, 4).to_mkldnn(), 2, stride={stride})
except (RuntimeError, ValueError) as error:
    assert "stride" in str(error).lower()
else:
    raise AssertionError("zero stride accepted")
''')

    def test_valid_cpu_operations_preserve_results(self):
        original = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)
        lengths = torch.tensor([3, 2])
        packed = torch.nn.utils.rnn.pack_padded_sequence(original, lengths)
        padded, restored_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed)
        torch.testing.assert_close(padded[:, 0], original[:, 0])
        torch.testing.assert_close(padded[:2, 1], original[:2, 1])
        torch.testing.assert_close(restored_lengths, lengths)
        self.assertTrue(torch.equal(padded[2, 1], torch.zeros(4)))
        for cell_type in (torch.nn.LSTMCell, torch.nn.GRUCell, torch.nn.RNNCell):
            cell = cell_type(4, 3)
            inputs = torch.zeros(2, 4)
            hidden = torch.zeros(2, 3)
            result = cell(inputs, (hidden, hidden) if cell_type is torch.nn.LSTMCell else hidden)
            outputs = result if isinstance(result, tuple) else (result,)
            for output in outputs:
                self.assertEqual(output.shape, hidden.shape)
                self.assertTrue(torch.isfinite(output).all())
        loss = torch.nn.functional.ctc_loss(
            torch.zeros(3, 1, 3).log_softmax(2), torch.tensor([1]), [3], [1]
        )
        self.assertTrue(torch.isfinite(loss))
        unit = torch.jit.CompilationUnit("def valid(x: int):\n    values: List[int] = [x]\n    return values[0]\n")
        self.assertEqual(unit.valid(23), 23)

    @unittest.skipUnless(torch.backends.mkldnn.is_available(), "MKLDNN is unavailable on this build")
    def test_valid_mkldnn_pooling_preserves_results(self):
        values = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        torch.testing.assert_close(
            torch.nn.functional.max_pool2d(values.to_mkldnn(), 2, stride=2).to_dense(),
            torch.tensor([[[[5.0, 7.0], [13.0, 15.0]]]]),
        )

    def test_profiler_future_retains_record_and_propagates_result_or_error(self):
        for legacy in (False, True):
            for failed, completed in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(legacy=legacy, failed=failed, completed=completed):
                    self.run_isolated(f'''
record = torch.ops.profiler.{'_record_function_enter' if legacy else '_record_function_enter_new'}("security-regression", None)
future = torch.futures.Future()
if {completed!r}:
    if {failed!r}:
        future.set_exception(ValueError("original failure"))
    else:
        future.set_result(19)
result = torch.ops.profiler._call_end_callbacks_on_jit_fut{'' if legacy else '._RecordFunction'}(record, future)
del record
if {failed!r}:
    if not {completed!r}:
        future.set_exception(ValueError("original failure"))
    try:
        result.wait()
    except (ValueError, RuntimeError) as error:
        assert "original failure" in str(error)
    else:
        raise AssertionError("future failure lost")
else:
    if not {completed!r}:
        future.set_result(19)
    assert result.wait() == 19
''')


class ExportTrustTests(unittest.TestCase):
    def test_export_rejects_executable_archive_payloads_without_side_effects(self):
        import torch._export

        with tempfile.TemporaryDirectory() as directory:
            marker = os.path.join(directory, "executed.txt")
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr("version", "1.0")
                archive.writestr("serialized_constants.pt", checkpoint(ExecutableState(marker)).getvalue())
                archive.writestr("serialized_state_dict.pt", checkpoint({}).getvalue())
                archive.writestr("serialized_exported_program.json",
                                 '{"expr_str": "__import__(\"os\").system(\"false\")"}')
            for loader in (torch.export.load, torch._export.load):
                buffer.seek(0)
                with self.assertRaisesRegex(ValueError, "trusted=True"):
                    loader(buffer)
                self.assertEqual(buffer.tell(), 0)
                self.assertFalse(os.path.exists(marker))

    def test_export_loading_requires_explicit_trust_before_reading_input(self):
        import torch._export

        for loader in (torch.export.load, torch._export.load):
            for trust in (False, None, 1, "true"):
                with self.subTest(loader=loader, trust=trust):
                    buffer = io.BytesIO(b"untrusted archive contents")
                    with self.assertRaisesRegex(ValueError, "trusted=True"):
                        loader(buffer, trusted=trust)
                    self.assertEqual(buffer.tell(), 0)
            with self.assertRaisesRegex(ValueError, "trusted=True"):
                loader("/nonexistent/untrusted-export.pt2")

    def test_authenticated_export_round_trip_preserves_inference_and_extra_files(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
                self.register_buffer("bias", torch.tensor([1.0, -1.0]))

            def forward(self, value):
                return value * self.weight + self.bias

        model = Model().eval()
        inputs = torch.tensor([4.0, 5.0])
        exported = torch.export.export(model, (inputs,))
        buffer = io.BytesIO()
        torch.export.save(exported, buffer, extra_files={"description.txt": "authenticated fixture"})
        buffer.seek(0)
        extra = {"description.txt": ""}
        restored = torch.export.load(buffer, trusted=True, extra_files=extra)
        torch.testing.assert_close(restored.module()(inputs), model(inputs))
        self.assertEqual(extra["description.txt"], "authenticated fixture")


class CompilerRegressionTests(unittest.TestCase):
    def test_symbolic_modular_indexing_does_not_assume_strict_positivity(self):
        import sympy
        from torch.utils._sympy.functions import ModularIndexing

        value = sympy.Symbol("value", integer=True, positive=True)
        expression = abs(ModularIndexing(value, 1, 1024) - 1)
        self.assertEqual(expression.subs(value, 1024), 1)
        self.assertEqual(expression.subs(value, 2), 1)

    def test_namedtuple_slices_preserve_python_tuple_semantics_when_compiled(self):
        inputs = torch.tensor([[2.0, 1.0], [4.0, 3.0]])
        for index in (slice(None, 2), slice(-2, None), slice(1, 1), slice(None, None, 2)):
            with self.subTest(index=index):
                def operation(value):
                    return torch.linalg.lu(value)[index]

                expected = operation(inputs)
                actual = torch.compile(operation, backend="aot_eager", fullgraph=True)(inputs)
                self.assertIs(type(actual), tuple)
                self.assertEqual(len(actual), len(expected))
                for result, reference in zip(actual, expected):
                    torch.testing.assert_close(result, reference)

    def test_fx_random_overloads_with_keyword_names_produce_executable_graphs(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        for operation in (lambda value: value.random_(0, 2), lambda value: value.uniform_(0, 2)):
            with self.subTest(operation=operation):
                graph = make_fx(operation)(torch.zeros(32))
                result = graph(torch.zeros(32))
                self.assertEqual(result.shape, (32,))
                self.assertTrue(torch.all((result >= 0) & (result < 2)).item())

    def test_compiled_like_factories_preserve_noncontiguous_layouts(self):
        from torch._inductor import config

        inputs = (
            torch.zeros(4, 5, 6).transpose(1, 2),
            torch.zeros(3, 4)[:, ::2].permute(1, 0),
            torch.rot90(torch.zeros(3, 4)),
        )

        operations = (
            ("full", lambda value: torch.full_like(value, 3)),
            ("uniform", torch.rand_like),
            ("normal", torch.randn_like),
            ("integer", lambda value: torch.randint_like(value, 1, 11)),
        )
        for name, operation in operations:
            compiled = torch.compile(operation, fullgraph=True)
            for value in inputs:
                with self.subTest(factory=name, shape=value.shape, stride=value.stride()):
                    with config.patch(fallback_random=True):
                        torch.manual_seed(1729)
                        expected = operation(value)
                        torch.manual_seed(1729)
                        actual = compiled(value)
                        if name == "normal" and value.numel() >= 16:
                            # Native contiguous and strided normal kernels use different RNG algorithms.
                            torch.manual_seed(1729)
                            torch.testing.assert_close(compiled(value), actual)
                            torch.manual_seed(1730)
                            self.assertFalse(torch.equal(compiled(value), actual))
                            self.assertGreater(actual.std().item(), 0)
                        else:
                            torch.testing.assert_close(actual, expected)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    if name == "uniform":
                        self.assertTrue(torch.all((actual >= 0) & (actual < 1)).item())
                    if name == "integer":
                        self.assertTrue(torch.all((actual >= 1) & (actual < 11)).item())

    def test_sparse_round_trip_matches_eager_compiled_execution(self):
        def operation(value):
            return value.to_sparse().to_dense() + 1

        value = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        torch.testing.assert_close(torch.compile(operation)(value), operation(value))


if __name__ == "__main__":
    unittest.main()
