"""Tests use only newly generated fixtures, never user models or data."""
import io
import importlib.util
import json
from pathlib import Path
import pickle
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile

import server


def string(value):
    raw = value.encode("utf-8")
    return b"X" + struct.pack("<I", len(raw)) + raw


def number(value):
    return b"J" + struct.pack("<i", value)


def sequence(values):
    return b"(" + b"".join(number(value) for value in values) + b"t"


def tensor(shape):
    # The actual PyTorch _rebuild_tensor_v2 layout; storage bytes aren't needed.
    return (b"ctorch._utils\n_rebuild_tensor_v2\n(" + b"(" + string("storage")
            + b"ctorch\nFloatStorage\n" + string("0") + string("cuda:0") + number(100) + b"tQ"
            + number(0) + sequence(shape) + sequence([1] * len(shape)) + b"\x89}tR")


def checkpoint():
    metadata = {
        "format_version": 2, "objective_version": 2,
        "objective_name": "baseline_residual_rubles_rms_scaled_mse_v2",
        "model_id": "torch_mlp_2_layers", "layers": [4, 3], "activation": "relu",
        "features": ["target_inflow_mean_3", "target_outflow_mean_3"],
        "feature_mean": [0.0, 0.0], "feature_scale": [1.0, 1.0],
        "active_features": [True, True], "residual_scale": [1.0, 1.0],
    }
    payload = pickle.dumps(metadata, protocol=2)[:-1]
    payload += string("state_dict") + b"}("
    for key, shape in [("0.weight", (4, 2)), ("0.bias", (4,)),
                       ("3.weight", (3, 4)), ("3.bias", (3,)),
                       ("6.weight", (2, 3)), ("6.bias", (2,))]:
        payload += string(key) + tensor(shape)
    return payload + b"us."


def zip_payload(payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/data/0", b"\x00" * 16)
        archive.writestr("archive/version", b"3\n")
    return buffer.getvalue()


class InspectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def inspect(self, payload):
        path = Path(self.temp.name) / "fixture.pt"
        path.write_bytes(payload)
        return server.inspect_file(path)

    def test_protocols_and_cycles(self):
        for protocol in range(min(pickle.HIGHEST_PROTOCOL, 5) + 1):
            data = {"features": ["a", "b"], "layers": (4, 3), "empty": None}
            parser = server.StaticPickle()
            self.assertEqual(parser.read(io.BytesIO(pickle.dumps(data, protocol=protocol))), data)
        circular = []
        circular.append(circular)
        root = server.StaticPickle().read(io.BytesIO(pickle.dumps(circular, protocol=2)))
        self.assertIs(root[0], root)
        self.assertEqual(server.collect_tensors(root)[0], [])

    def test_checkpoint_shapes_and_contract(self):
        report = self.inspect(zip_payload(checkpoint()))
        self.assertEqual(report["status"], "inspected", report)
        self.assertEqual(report["recognized_contract"], "cashgap_monthly_v2")
        self.assertEqual(report["linear_chain_hypothesis"]["widths"], [2, 4, 3, 2])
        self.assertEqual(len(report["tensors"]), 6)
        self.assertEqual(report["tensors"][0]["saved_device"], "cuda:0")
        self.assertEqual(report["consistency_issues"], [])
        self.assertEqual(report["metadata"]["feature_mean"]["values"], "not included")
        self.assertIn("ИНН", server.markdown_report(report))

    def test_legacy_format(self):
        payload = (pickle.dumps(server.MAGIC, protocol=2) + pickle.dumps(1001, protocol=2)
                   + pickle.dumps({"little_endian": True}, protocol=2) + checkpoint() + b"raw storage")
        report = self.inspect(payload)
        self.assertEqual(report["container"], "legacy_torch_save")
        self.assertEqual(report["status"], "inspected")

    def test_reduce_does_not_execute(self):
        marker = Path(self.temp.name) / "MUST_NOT_EXIST"
        command = "touch " + str(marker)
        payload = b"\x80\x02cos\nsystem\n" + string(command) + b"\x85R."
        report = self.inspect(zip_payload(payload))
        self.assertEqual(report["status"], "inspected")
        self.assertIn("os.system", report["referenced_globals"])
        self.assertFalse(marker.exists())
        self.assertNotIn(command, json.dumps(report))

    def test_raw_state_dict_does_not_invent_business_meaning(self):
        payload = b"\x80\x02}" + string("0.weight") + tensor((2, 73)) + b"s."
        report = self.inspect(zip_payload(payload))
        self.assertEqual(report["checkpoint_kind"], "state_dict_only")
        self.assertIsNone(report["recognized_contract"])
        self.assertIsNone(report["feature_count"])
        self.assertEqual(report["linear_chain_hypothesis"]["widths"], [73, 2])

    def test_bad_file_produces_report(self):
        report = self.inspect(b"not a checkpoint")
        self.assertEqual(report["status"], "partial")
        self.assertIn("error", report)
        self.assertIn("частичный", server.markdown_report(report))

    def test_nested_object_state_and_ordered_dict(self):
        payload = (b"\x80\x02ctorch.nn.modules.linear\nLinear\n)\x81}"
                   + string("_parameters") + b"ccollections\nOrderedDict\n)R"
                   + string("weight") + tensor((2, 4)) + b"ssb.")
        report = self.inspect(zip_payload(payload))
        self.assertEqual(report["status"], "inspected", report)
        self.assertEqual(report["tensors"][0]["shape"], [2, 4])


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.http = server.DiagnosticServer(("127.0.0.1", 0), self.temp.name, 1024 * 1024, 20)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:{}".format(self.http.server_port)

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=5)
        self.temp.cleanup()

    def request(self, route, data=None, token=True, **headers):
        if token:
            headers["X-Diagnostic-Token"] = self.http.token
        request = urllib.request.Request(self.base + route, data=data, headers=headers)
        return urllib.request.urlopen(request, timeout=30)

    def test_auth_health_and_real_worker_upload(self):
        with self.request("/", token=False) as response:
            self.assertIn("Диагностика".encode(), response.read())
        with self.assertRaises(urllib.error.HTTPError) as failure:
            self.request("/api/health", token=False)
        self.assertEqual(failure.exception.code, 403)
        failure.exception.close()
        with self.request("/api/health") as response:
            self.assertFalse(json.load(response)["torch_required"])
        with self.request("/api/inspect", zip_payload(checkpoint()), **{"X-Filename": "my_model.pt"}) as response:
            result = json.load(response)
        self.assertEqual(result["report"]["status"], "inspected")
        self.assertEqual(result["report"]["file"]["name"], "my_model.pt")
        self.assertEqual(len(list(Path(self.temp.name).glob("*.md"))), 1)
        self.assertEqual(len(list(Path(self.temp.name).glob("*.json"))), 1)
        self.assertEqual(len(list(Path(self.temp.name).glob("*.pt"))), 0)

    def test_origin_size_busy_and_malformed(self):
        # Oversize is rejected using Content-Length without sending a huge body.
        for data, headers, code in [(b"x", {"Origin": "http://evil.example"}, 403),
                                     (b"x", {"Content-Length": str(1024 * 1024 + 1)}, 413)]:
            with self.assertRaises(urllib.error.HTTPError) as failure:
                self.request("/api/inspect", data, **headers)
            self.assertEqual(failure.exception.code, code)
            failure.exception.close()
        self.http.slot.acquire()
        try:
            with self.assertRaises(urllib.error.HTTPError) as failure:
                self.request("/api/inspect", b"x")
            self.assertEqual(failure.exception.code, 409)
            failure.exception.close()
        finally:
            self.http.slot.release()
        with self.request("/api/inspect", b"not pickle") as response:
            self.assertEqual(json.load(response)["report"]["status"], "partial")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional real PyTorch fixture tests; server itself needs no torch")
class RealTorchTests(unittest.TestCase):
    def test_new_generated_files_only(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(73, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
        with tempfile.TemporaryDirectory() as folder:
            for legacy, protocol in [(False, 2), (False, 4), (False, 5), (True, 2)]:
                path = Path(folder) / "generated.pt"
                torch.save({"state_dict": model.state_dict(), "layers": [8]}, path,
                           pickle_protocol=protocol, _use_new_zipfile_serialization=not legacy)
                report = server.inspect_file(path)
                self.assertEqual(report["status"], "inspected", report)
                self.assertEqual(report["linear_chain_hypothesis"]["widths"], [73, 8, 2])
            if importlib.util.find_spec("numpy"):
                import numpy as np
                path = Path(folder) / "numpy_metadata.pt"
                torch.save({"state_dict": model.state_dict(), "feature_mean": np.zeros(73),
                            "feature_scale": np.ones(73), "features": ["x{}".format(i) for i in range(73)]}, path)
                report = server.inspect_file(path)
                self.assertEqual(report["status"], "inspected", report)
                self.assertEqual(report["metadata"]["feature_mean"]["shape"], [73])
                self.assertEqual(report["consistency_issues"], [])
            path = Path(folder) / "whole_object.pt"
            torch.save(model, path)
            report = server.inspect_file(path)
            self.assertEqual(report["status"], "inspected", report)
            self.assertEqual(len(report["tensors"]), 4)
            path = Path(folder) / "torchscript.pt"
            torch.jit.trace(model, torch.zeros(1, 73)).save(str(path))
            report = server.inspect_file(path)
            self.assertEqual(report["status"], "inspected", report)
            self.assertEqual(report["container"], "torchscript_zip_candidate")


if __name__ == "__main__":
    unittest.main(verbosity=2)
