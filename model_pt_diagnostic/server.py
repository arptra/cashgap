#!/usr/bin/env python3
"""Standalone, offline .pt metadata inspector. Python 3.8+, standard library only.

No torch.load, pickle.load, imports from checkpoints, or model execution.
Pickle opcodes are interpreted as inert descriptions, not Python instructions.
"""
import sys

if sys.version_info < (3, 8):
    raise SystemExit("Нужен Python 3.8 или новее. Внешние библиотеки не нужны.")

import argparse
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import pickletools
import re
import secrets
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit
import uuid
import zipfile

VERSION = "1.0"
PICKLE_LIMIT = 16 * 1024 * 1024
OP_LIMIT = 300_000
WALK_LIMIT = 30_000
MARK = object()
MAGIC = 0x1950A86A20F9469CFC6C


class Symbol:
    def __init__(self, name, args=()):
        self.name = name
        self.args = args
        self.state = None


class ParseProblem(Exception):
    pass


def label(value):
    return value.name if isinstance(value, Symbol) else str(value)[:300]


class StaticPickle:
    """A bounded symbolic stack machine. Never resolve/call a pickle GLOBAL."""

    def __init__(self):
        self.globals = set()
        self.operations = 0
        self.protocol = 0

    def read(self, stream):
        stack, memo = [], {}

        def marked():
            for index in range(len(stack) - 1, -1, -1):
                if stack[index] is MARK:
                    result = stack[index + 1:]
                    del stack[index:]
                    return result
            raise ParseProblem("Не найден MARK в pickle.")

        def pairs(items):
            if len(items) % 2:
                raise ParseProblem("Нечётное число элементов словаря.")
            return zip(items[::2], items[1::2])

        for opcode, arg, position in pickletools.genops(stream):
            self.operations += 1
            if self.operations > OP_LIMIT or len(stack) > 100_000 or len(memo) > 100_000:
                raise ParseProblem("Достигнут лимит сложности pickle.")
            name = opcode.name
            if name == "PROTO":
                self.protocol = max(self.protocol, arg)
            elif name == "FRAME":
                pass
            elif name == "MARK":
                stack.append(MARK)
            elif name == "STOP":
                if len(stack) != 1:
                    raise ParseProblem("Некорректный стек в конце pickle.")
                return stack.pop()
            elif name in {"NONE", "NEWTRUE", "NEWFALSE"}:
                stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[name])
            elif name in {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4",
                          "FLOAT", "BINFLOAT", "STRING", "BINSTRING", "SHORT_BINSTRING",
                          "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
                          "BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8"}:
                stack.append(arg)
            elif name in {"EMPTY_LIST", "EMPTY_DICT", "EMPTY_TUPLE", "EMPTY_SET"}:
                stack.append({"EMPTY_LIST": list, "EMPTY_DICT": dict,
                              "EMPTY_TUPLE": tuple, "EMPTY_SET": set}[name]())
            elif name in {"LIST", "TUPLE", "FROZENSET", "DICT"}:
                items = marked()
                stack.append(dict(pairs(items)) if name == "DICT" else
                             {"LIST": list, "TUPLE": tuple, "FROZENSET": frozenset}[name](items))
            elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
                count = int(name[-1])
                items = stack[-count:]
                del stack[-count:]
                stack.append(tuple(items))
            elif name == "APPEND":
                value = stack.pop()
                stack[-1].append(value)
            elif name in {"APPENDS", "ADDITEMS", "SETITEMS"}:
                items = marked()
                target = stack[-1]
                if name == "APPENDS":
                    target.extend(items)
                elif name == "ADDITEMS":
                    target.update(items)
                else:
                    target.update(pairs(items))
            elif name == "SETITEM":
                value, key = stack.pop(), stack.pop()
                stack[-1][key] = value
            elif name in {"PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE"}:
                memo[len(memo) if name == "MEMOIZE" else int(arg)] = stack[-1]
            elif name in {"GET", "BINGET", "LONG_BINGET"}:
                stack.append(memo[int(arg)])
            elif name == "POP":
                stack.pop()
            elif name == "POP_MARK":
                marked()
            elif name == "DUP":
                stack.append(stack[-1])
            elif name in {"GLOBAL", "STACK_GLOBAL"}:
                if name == "GLOBAL":
                    module, member = arg.split(" ", 1)
                else:
                    member, module = stack.pop(), stack.pop()
                    if not isinstance(member, str) or not isinstance(module, str):
                        raise ParseProblem("Некорректный STACK_GLOBAL.")
                reference = module + "." + member
                self.globals.add(reference)
                stack.append(Symbol(reference))
            elif name in {"EXT1", "EXT2", "EXT4"}:
                reference = "pickle_extension_{}".format(arg)
                self.globals.add(reference)
                stack.append(Symbol(reference))
            elif name in {"REDUCE", "NEWOBJ", "NEWOBJ_EX"}:
                kwargs = stack.pop() if name == "NEWOBJ_EX" else None
                arguments, function = stack.pop(), stack.pop()
                reference = label(function)
                if name == "REDUCE" and reference == "collections.OrderedDict":
                    # Use an ordinary inert mapping, not the referenced callable.
                    stack.append(dict(arguments[0]) if arguments else {})
                else:
                    node = Symbol(reference, arguments)
                    if kwargs is not None:
                        node.state = {"constructor_keywords": kwargs}
                    stack.append(node)
            elif name == "BUILD":
                state = stack.pop()
                if isinstance(stack[-1], Symbol):
                    stack[-1].state = state
                elif not isinstance(stack[-1], dict):
                    raise ParseProblem("BUILD для неподдерживаемого контейнера.")
                # OrderedDict._metadata isn't needed for weight-shape inspection.
            elif name in {"BINPERSID", "PERSID"}:
                stack.append(Symbol("persistent_storage", (stack.pop() if name == "BINPERSID" else arg,)))
            elif name in {"INST", "OBJ"}:
                items = marked()
                if name == "INST":
                    reference = arg.replace(" ", ".", 1)
                    self.globals.add(reference)
                else:
                    reference, items = label(items[0]), items[1:]
                stack.append(Symbol(reference, tuple(items)))
            else:
                raise ParseProblem("Статический анализ не поддерживает opcode {} (позиция {}).".format(name, position))
        raise ParseProblem("Нет завершающего STOP.")


def dimensions(value):
    if isinstance(value, (tuple, list)) and len(value) <= 32:
        if all(type(item) is int and 0 <= item <= 10**12 for item in value):
            return list(value)
    return None


def tensor_info(value):
    if not isinstance(value, Symbol):
        return None
    args = value.args
    if value.name in {"torch._utils._rebuild_parameter", "torch._utils._rebuild_parameter_with_state"}:
        return tensor_info(args[0]) if args else None
    if value.name in {"torch._utils._rebuild_tensor", "torch._utils._rebuild_tensor_v2",
                      "torch._utils._rebuild_tensor_v3"} and len(args) >= 4:
        shape = dimensions(args[2])
        if shape is None:
            return None
        result = {"shape": shape, "elements": math.prod(shape), "dtype": "unknown", "saved_device": "unknown"}
        storage = args[0]
        if isinstance(storage, Symbol) and storage.name == "persistent_storage":
            data = storage.args[0]
            if isinstance(data, (tuple, list)) and len(data) >= 4:
                result["dtype"] = label(data[1])
                result["saved_device"] = label(data[3])
        if value.name.endswith("_v3") and len(args) > 6:
            result["dtype"] = label(args[6])
        return result
    return None


def describe(value, depth=0):
    """Metadata only: never report tensor/array payloads or arbitrary large text."""
    tensor = tensor_info(value)
    if tensor:
        return {"type": "tensor", **tensor}
    if isinstance(value, Symbol):
        result = {"type": "symbolic_object", "class_or_factory": value.name}
        if value.name.endswith("._reconstruct") and isinstance(value.state, tuple) and len(value.state) > 1:
            result["shape"] = dimensions(value.state[1])
        return result
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "length": len(value), "values": "not included"}
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value[:500] if isinstance(value, str) else value
    if depth >= 4:
        return {"type": type(value).__name__, "details": "depth limit"}
    if isinstance(value, dict):
        return {label(key): describe(item, depth + 1) for key, item in list(value.items())[:100]}
    if isinstance(value, (list, tuple)):
        return [describe(item, depth + 1) for item in value[:2000]]
    return {"type": type(value).__name__}


def collect_tensors(root):
    results, seen = [], set()
    pending, walked = [("$", root, 0)], 0
    while pending and walked < WALK_LIMIT:
        path, value, depth = pending.pop()
        walked += 1
        tensor = tensor_info(value)
        if tensor:
            results.append({"path": path, **tensor})
            continue
        if depth >= 40 or id(value) in seen:
            continue
        if isinstance(value, (dict, list, tuple, Symbol)):
            seen.add(id(value))
        children = []
        if isinstance(value, dict):
            children = [(path + "." + label(key), item, depth + 1) for key, item in value.items()]
        elif isinstance(value, (tuple, list)):
            children = [(path + "[{}]".format(index), item, depth + 1) for index, item in enumerate(value)]
        elif isinstance(value, Symbol):
            children = [(path + ".state", value.state, depth + 1), (path + ".args", value.args, depth + 1)]
        pending.extend(reversed(children[:WALK_LIMIT]))
    return results, bool(pending)


META_KEYS = ("format_version", "model_id", "model_name", "objective_version", "objective_name",
             "features", "feature_names", "input_features", "layers", "hidden_layers", "activation",
             "dropout", "horizon", "horizon_days", "target_names", "output_names", "outputs",
             "feature_mean", "feature_scale", "active_features", "residual_scale", "target_scale",
             "target_mean", "scaler", "config", "model_config", "hyper_parameters")

NORMALIZATION_KEYS = {"feature_mean", "feature_scale", "active_features", "residual_scale",
                      "target_scale", "target_mean", "scaler"}


def metadata_value(key, value, depth=0):
    if key in NORMALIZATION_KEYS and isinstance(value, (list, tuple)):
        return {"type": "sequence", "shape": [len(value)], "values": "not included"}
    if key in {"config", "model_config", "hyper_parameters"}:
        allowed = set(META_KEYS) | {"learning_rate", "weight_decay", "batch_size", "epochs",
                                    "input_size", "input_dim", "output_size", "output_dim",
                                    "hidden_size", "num_layers", "lookback", "context_length"}
        if isinstance(value, dict) and depth < 2:
            return {name: metadata_value(name, item, depth + 1)
                    for name, item in value.items() if name in allowed}
        return {"type": type(value).__name__, "details": "custom configuration not expanded"}
    return describe(value)


def analyse_root(report, root, parser):
    tensors, truncated = collect_tensors(root)
    report.update(root_type=type(root).__name__, pickle_protocol=parser.protocol,
                  pickle_operations=parser.operations, referenced_globals=sorted(parser.globals),
                  tensors=tensors, traversal_truncated=truncated)
    top = root if isinstance(root, dict) else {}
    report["top_level_fields"] = [{"name": label(key), "type": type(value).__name__}
                                  for key, value in list(top.items())[:200]]
    report["metadata"] = {key: metadata_value(key, top[key]) for key in META_KEYS if key in top}
    metadata = report["metadata"]
    feature_list = next((top.get(key) for key in ("features", "feature_names", "input_features")
                         if isinstance(top.get(key), (tuple, list))
                         and all(isinstance(item, str) for item in top[key])), None)
    report["feature_count"] = len(feature_list) if feature_list is not None else None
    state = next((key for key in ("state_dict", "model_state_dict", "model", "weights", "net")
                  if isinstance(top.get(key), dict)), None)
    if state:
        report["checkpoint_kind"] = "checkpoint_with_state_dict"
        weights = [item for item in tensors if item["path"].startswith("$." + state + ".")]
    elif top and tensors and all(tensor_info(value) is not None for value in top.values()):
        report["checkpoint_kind"] = "state_dict_only"
        weights = tensors
    elif isinstance(root, Symbol):
        report["checkpoint_kind"] = "serialized_object_description"
        weights = tensors
    else:
        report["checkpoint_kind"] = "generic_checkpoint_or_data"
        weights = []
    report["state_dict_field"] = state
    report["weight_tensor_elements"] = sum(item["elements"] for item in weights)
    matrices = [item for item in weights if len(item["shape"]) == 2 and item["path"].endswith(".weight")]
    chain = [matrices[0]["shape"][1]] + [item["shape"][0] for item in matrices] if matrices else []
    compatible = bool(matrices) and all(left["shape"][0] == right["shape"][1]
                                        for left, right in zip(matrices, matrices[1:]))
    report["linear_chain_hypothesis"] = {
        "compatible_dimensions": compatible,
        "widths": chain if compatible else [],
        "candidate_weight_paths": [item["path"] for item in matrices],
        "warning": "Гипотеза по формам и порядку матриц. Это не восстановленный forward: матрица может быть embedding, а модель — иметь ветви.",
    }
    declared_layers = dimensions(top.get("layers"))
    issues = []
    if feature_list is not None and compatible and len(feature_list) != chain[0]:
        issues.append("Число названий признаков не совпадает с входом предполагаемой цепочки матриц.")
    if declared_layers is not None and compatible and declared_layers != chain[1:-1]:
        issues.append("Заявленные скрытые слои не совпадают с предполагаемой цепочкой весов.")
    for key in ("feature_mean", "feature_scale", "active_features"):
        if key in top:
            info = describe(top[key])
            shape = info.get("shape") if isinstance(info, dict) else [len(info)] if isinstance(info, list) else None
            if shape is not None and feature_list is not None and shape != [len(feature_list)]:
                issues.append("Размер {} не совпадает с числом признаков.".format(key))
    report["consistency_issues"] = issues
    known = (top.get("format_version") == 2 and top.get("objective_version") == 2
             and top.get("objective_name") == "baseline_residual_rubles_rms_scaled_mse_v2"
             and feature_list is not None
             and {"target_inflow_mean_3", "target_outflow_mean_3"}.issubset(feature_list))
    report["recognized_contract"] = "cashgap_monthly_v2" if known else None
    findings = []
    if known:
        findings.append("Метаданные соответствуют месячному формату Cashgap v2: прогноз зачислений и списаний через поправку к среднему. Это распознавание описания, не проверка работоспособности весов.")
        findings.append("Одного ИНН недостаточно: нужны история клиента и подготовка признаков. ИНН используется для поиска данных вне нейросети.")
    elif feature_list is not None:
        findings.append("Найден список входных признаков. Он включён в отчёт; их смысл и порядок подготовки нужно подтвердить по коду обучения.")
    else:
        findings.append("Список входных признаков не найден в известных полях. По одним размерам весов нельзя узнать, какие данные передавать модели.")
    if compatible:
        findings.append("Матрицы весов совместимы с цепочкой {}. Гипотеза о скрытых слоях: {}. Это не доказательство архитектуры.".format(
            " → ".join(map(str, chain)), len(chain) - 2))
    if "activation" not in metadata:
        findings.append("Активация не найдена в стандартном поле: в state_dict функции ReLU/Dropout обычно не представлены весами.")
    if not tensors:
        findings.append("Поддерживаемые описания тензоров не обнаружены. Это не доказывает, что весов в файле нет.")
    findings.append("Предсказание не выполнялось. Точность, пригодность модели и достаточность одного ИНН для неизвестного формата этим отчётом не подтверждаются.")
    report["findings_ru"] = findings
    report["needed_next_ru"] = [
        "Для следующего разбора передайте model_report.md и model_report.json после проверки их содержимого.",
        "Нужен точный контракт входа: названия, порядок, единицы измерения и расчёт признаков на дату прогноза.",
        "Нужны архитектура/forward и преобразования входов и выходов, если их нельзя восстановить из метаданных.",
        "Для персонального прогноза нужен источник актуальных признаков или истории клиента по ИНН.",
        "Для проверки запуска полезен эталонный пример входа и ожидаемого выхода от автора модели.",
    ]
    if truncated:
        report["warnings_ru"].append("Обход ограничен по объёму; инвентаризация тензоров может быть неполной.")


def inspect_file(path, original_name=None):
    path = Path(path)
    report = {
        "report_version": VERSION,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "partial",
        "file": {"name": original_name or path.name, "bytes": path.stat().st_size},
        "environment": {"python": sys.version.split()[0], "os": sys.platform},
        "inspection_mode": "static_pickle_metadata_no_execution",
        "warnings_ru": [
            "Это статический осмотр: модель, CUDA и пользовательские классы не запускались.",
            "Числовые значения весов, массивов и примеров клиентов не включены. Имена полей, классов, признаков и разрешённые метаданные включены: проверьте отчёт перед передачей.",
            "Повреждение содержимого хранилищ весов и их численная корректность не проверялись.",
        ],
    }
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    report["file"]["sha256"] = digest.hexdigest()
    parser = StaticPickle()
    try:
        if zipfile.is_zipfile(str(path)):
            with zipfile.ZipFile(str(path)) as archive:
                members = archive.infolist()
                if len(members) > 50_000:
                    raise ParseProblem("Слишком много записей ZIP.")
                names = [item.filename for item in members]
                report["container"] = "torch_zip_candidate"
                report["archive"] = {"entries": len(members), "uncompressed_bytes": sum(item.file_size for item in members),
                                     "entry_names_sample": names[:100]}
                script = any("/code/" in "/" + name for name in names) and any(name.endswith("constants.pkl") for name in names)
                if script:
                    report["container"] = "torchscript_zip_candidate"
                    report["warnings_ru"].append("Похоже на TorchScript. Код архива не выполнялся; восстановление сигнатуры forward этим инструментом не поддерживается.")
                candidates = [item for item in members if item.filename == "data.pkl" or item.filename.endswith("/data.pkl")]
                if len(candidates) != 1:
                    raise ParseProblem("Ожидался один data.pkl; найдено {}. Формат может отличаться от torch.save.".format(len(candidates)))
                entry = candidates[0]
                if entry.file_size > PICKLE_LIMIT:
                    raise ParseProblem("data.pkl больше лимита 16 МБ. Анализ остановлен без распаковки весов.")
                with archive.open(entry) as stream:
                    payload = stream.read(PICKLE_LIMIT + 1)
                if len(payload) > PICKLE_LIMIT:
                    raise ParseProblem("Превышен лимит размера pickle.")
                root = parser.read(io.BytesIO(payload))
        else:
            with path.open("rb") as source:
                stream = io.BytesIO(source.read(PICKLE_LIMIT))
            root = parser.read(stream)
            if type(root) is int and root == MAGIC:
                report["container"] = "legacy_torch_save"
                report["legacy_serialization_version"] = describe(parser.read(stream))
                report["legacy_system_info"] = describe(parser.read(stream))
                root = parser.read(stream)
            else:
                report["container"] = "plain_pickle_candidate"
                report["warnings_ru"].append("Это не обычный ZIP torch.save; расширение .pt само по себе не подтверждает формат модели.")
        analyse_root(report, root, parser)
        report["status"] = "inspected"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        report["referenced_globals"] = sorted(parser.globals)
        report["findings_ru"] = [
            "Полный статический разбор не завершён. Это может быть иной формат, неподдерживаемая конструкция, лимит или повреждённый файл.",
            "Передайте этот частичный отчёт: причина и найденные ссылки на классы помогут выбрать следующий способ проверки.",
        ]
    return report


def md_text(value):
    return str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|").replace("`", "'")[:1000]


def markdown_report(report):
    lines = ["# Диагностика файла модели", "", "## Файл и результат", "",
             "- Файл: `{}`".format(md_text(report["file"]["name"])),
             "- Размер: {:,} байт.".format(report["file"]["bytes"]),
             "- SHA-256: `{}`".format(report["file"]["sha256"]),
             "- Статус: {}.".format("структура прочитана" if report["status"] == "inspected" else "частичный отчёт"),
             "- Контейнер: `{}`.".format(report.get("container", "не определён")),
             "- Режим: статический разбор без запуска модели; это НЕ прогноз и НЕ проверка точности.",
             "", "## Что удалось установить", ""]
    lines.extend("- " + item for item in report.get("findings_ru", []))
    if report.get("error"):
        lines.extend(["", "## Почему разбор ограничен", "", md_text(report["error"])])
    if report.get("consistency_issues"):
        lines.extend(["", "## Несоответствия", ""])
        lines.extend("- " + item for item in report["consistency_issues"])
    lines.extend(["", "## Найденные метаданные", ""])
    for key, value in report.get("metadata", {}).items():
        if key in {"features", "feature_names", "input_features"} and isinstance(value, list):
            lines.extend(["", "### Признаки: {} (порядок сохранён)".format(len(value)), ""])
            lines.extend("{}. `{}`".format(index + 1, md_text(item)) for index, item in enumerate(value))
        else:
            lines.append("- `{}`: `{}`".format(key, md_text(json.dumps(value, ensure_ascii=False))))
    lines.extend(["", "## Тензоры — формы, без значений весов", "",
                  "Число элементов — не обязательно число обучаемых параметров: здесь могут быть буферы и повторные ссылки.", "",
                  "| Поле | Размеры | Элементов | Тип | Сохранённое устройство |",
                  "|---|---|---:|---|---|"])
    for item in report.get("tensors", [])[:1000]:
        lines.append("| {} | {} | {} | {} | {} |".format(*(md_text(item[key]) for key in
                     ("path", "shape", "elements", "dtype", "saved_device"))))
    if len(report.get("tensors", [])) > 1000:
        lines.extend(["", "Показаны первые 1000 записей; остальные — в JSON."])
    lines.extend(["", "## Ссылки на классы и функции в файле", "",
                  "Это ссылки из pickle, а не список реально импортированных библиотек. Они не исполнялись.", ""])
    lines.extend("- `{}`".format(md_text(item)) for item in report.get("referenced_globals", []))
    lines.extend(["", "## Что нужно дальше", ""])
    lines.extend("- " + item for item in report.get("needed_next_ru", []))
    lines.extend(["", "## Ограничения и конфиденциальность", ""])
    lines.extend("- " + item for item in report.get("warnings_ru", []))
    return "\n".join(lines) + "\n"


PAGE = r'''<!doctype html>
<html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Диагностика model.pt</title>
<style>
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#18232c;background:#edf2f5}
body{max-width:960px;margin:40px auto;padding:0 20px}h1{font-size:32px;letter-spacing:-1px}
.card{background:white;padding:26px;border:1px solid #dce3e8;border-radius:18px;margin:20px 0}
.muted{color:#576774;line-height:1.6}button{background:#076b65;color:white;border:0;padding:12px 20px;border-radius:9px;font:inherit;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}input{display:block;margin:18px 0;max-width:100%}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.65 ui-monospace,monospace;max-height:650px;overflow:auto}
.actions{display:flex;flex-wrap:wrap;gap:10px}.badge{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:#076b65}
#status{white-space:pre-wrap;line-height:1.5;margin-top:18px}progress{width:100%;margin-top:18px}#error{color:#a22621}
</style>
<div class="badge">Локальная диагностика · без PyTorch и CUDA</div>
<h1>Что находится внутри model.pt?</h1>
<p class="muted">Загрузите файл, получите описание структуры и скачайте отчёт для разбора. Модель не запускается, предсказания не генерируются.</p>
<section class="card"><label for="file">Файл модели (.pt, .pth или .ckpt)</label>
<input id="file" type="file" accept=".pt,.pth,.ckpt,.bin,.pkl"><button id="inspect">Проверить файл</button>
<progress id="progress" max="100" value="0" hidden></progress><div id="status" role="status">Выберите файл на этом компьютере.</div><p id="error"></p></section>
<section class="card" id="result" hidden><h2>Отчёт готов</h2><p class="muted">Передайте оба файла: Markdown для чтения, JSON для подробного разбора. Сначала проверьте, можно ли передавать названия признаков и метаданные.</p>
<div class="actions"><button id="md">Скачать model_report.md</button><button id="json">Скачать model_report.json</button><button id="copy">Копировать отчёт</button></div><pre id="report"></pre></section>
<p class="muted">Файл обрабатывается только на ПК, где запущен сервер, и удаляется из временного каталога после обработки. Отчёты остаются в папке reports. Внешних запросов и аналитики нет.</p>
<script>
const $=id=>document.getElementById(id);
const hash=new URLSearchParams(location.hash.slice(1));
if(hash.has('token')){sessionStorage.setItem('model_diag_token',hash.get('token'));history.replaceState(null,'',location.pathname);}
const token=sessionStorage.getItem('model_diag_token')||'';
let current=null;
if(!token){$('error').textContent='Откройте полную ссылку с #token= из консоли сервера.';}
function download(name,text){const a=document.createElement('a');const url=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('md').onclick=()=>download('model_report.md',current.markdown);
$('json').onclick=()=>download('model_report.json',JSON.stringify(current.report,null,2));
$('copy').onclick=async()=>{try{await navigator.clipboard.writeText(current.markdown);$('status').textContent='Отчёт скопирован.';}catch(e){$('error').textContent='Копирование недоступно. Скачайте файл Markdown.';}};
$('inspect').onclick=async()=>{
 const f=$('file').files[0];if(!f){$('error').textContent='Сначала выберите файл.';return;}
 $('error').textContent='';$('result').hidden=true;$('inspect').disabled=true;$('progress').hidden=false;$('progress').value=0;
 try{const response=await fetch('/api/health',{headers:{'X-Diagnostic-Token':token}});const health=await response.json();if(!response.ok)throw Error(health.error||'Сервер недоступен');if(f.size===0||f.size>health.max_upload_bytes)throw Error('Пустой файл или превышен лимит '+Math.round(health.max_upload_bytes/1024/1024)+' МБ.');}
 catch(e){$('error').textContent=e.message;$('inspect').disabled=false;$('progress').hidden=true;return;}
 $('status').textContent='Загрузка файла на диагностический сервер…';
 const xhr=new XMLHttpRequest();xhr.open('POST','/api/inspect');xhr.timeout=300000;
 xhr.setRequestHeader('X-Diagnostic-Token',token);xhr.setRequestHeader('X-Filename',encodeURIComponent(f.name));xhr.setRequestHeader('Content-Type','application/octet-stream');
 xhr.upload.onprogress=e=>{if(e.lengthComputable){$('progress').value=100*e.loaded/e.total;if(e.loaded===e.total){$('status').textContent='Файл загружен. Выполняется статический разбор в отдельном процессе…';$('progress').removeAttribute('value');}}};
 xhr.onload=()=>{try{const r=JSON.parse(xhr.responseText);if(xhr.status!==200)throw Error(r.error||'Ошибка сервера');current=r;$('report').textContent=r.markdown;$('result').hidden=false;$('status').textContent=r.report.status==='inspected'?'Структура прочитана. Это ещё не проверка запуска модели.':'Получен частичный отчёт — его тоже можно передать для разбора.';}catch(e){$('error').textContent=e.message;}};
 xhr.onerror=()=>{$('error').textContent='Соединение прервано. Проверьте консоль сервера.';};
 xhr.ontimeout=()=>{$('error').textContent='Истекло время ожидания. Проверьте консоль и папку reports.';};
 xhr.onloadend=()=>{$('inspect').disabled=false;$('progress').hidden=true;};xhr.send(f);
};
</script></html>'''


class DiagnosticServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, report_dir, max_bytes, timeout):
        super().__init__(address, Handler)
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.worker_timeout = timeout
        self.token = secrets.token_urlsafe(32)
        self.slot = threading.BoundedSemaphore(1)


class Handler(BaseHTTPRequestHandler):
    server_version = "ModelDiagnostic/" + VERSION

    def setup(self):
        super().setup()
        self.connection.settimeout(60)

    def log_message(self, fmt, *args):
        # Never log URL fragments, query strings, authentication headers, or filenames.
        print("[HTTP] {} {}".format(self.command, getattr(self, "response_status", "-")), flush=True)

    def send(self, status, payload, content_type="application/json; charset=utf-8"):
        self.response_status = status
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def authenticated(self):
        supplied = self.headers.get("X-Diagnostic-Token", "")
        if not secrets.compare_digest(supplied.encode(), self.server.token.encode()):
            self.send(403, {"error": "Нет доступа. Откройте ссылку с токеном из консоли сервера."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            self.send(403, {"error": "Запрос с другого сайта запрещён."})
            return False
        return True

    def do_GET(self):
        route = urlsplit(self.path).path
        if route == "/":
            self.send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif route == "/api/health" and self.authenticated():
            self.send(200, {"status": "ok", "version": VERSION, "python": sys.version.split()[0],
                            "max_upload_bytes": self.server.max_bytes, "torch_required": False})
        elif route != "/api/health":
            self.send(404, {"error": "Страница не найдена."})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/inspect":
            self.send(404, {"error": "Маршрут не найден."})
            return
        if not self.authenticated():
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Нужен Content-Length; chunked upload не поддерживается.")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= self.server.max_bytes:
                self.send(413, {"error": "Размер должен быть от 1 байта до {} МБ.".format(self.server.max_bytes // (1024**2))})
                return
        except ValueError as exc:
            self.send(400, {"error": str(exc)})
            return
        if not self.server.slot.acquire(blocking=False):
            self.send(409, {"error": "Уже обрабатывается файл. Дождитесь завершения и повторите."})
            return
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:10]
        try:
            filename = unquote(self.headers.get("X-Filename", "model.pt")).replace("\\", "/").split("/")[-1][:160]
            with tempfile.TemporaryDirectory(prefix="model_pt_inspect_") as folder:
                uploaded, result_path = Path(folder) / "upload.pt", Path(folder) / "result.json"
                remaining, deadline = size, time.monotonic() + 120
                with uploaded.open("wb") as output:
                    while remaining:
                        if time.monotonic() > deadline:
                            raise TimeoutError("Загрузка заняла более 120 секунд.")
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Файл загружен не полностью.")
                        output.write(chunk)
                        remaining -= len(chunk)
                print("[{}] Получено {} байт. Статический разбор…".format(run_id, size), flush=True)
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", str(uploaded), str(result_path)]
                result = subprocess.run(command, capture_output=True, timeout=self.server.worker_timeout)
                if result.returncode != 0 or not result_path.is_file():
                    report = inspect_failure(uploaded, filename, "Процесс диагностики завершился с кодом {}. {}".format(
                        result.returncode, result.stderr.decode("utf-8", errors="replace")[-2000:]))
                else:
                    report = json.loads(result_path.read_text(encoding="utf-8"))
                    report["file"]["name"] = filename
                self.finish_report(run_id, report)
        except subprocess.TimeoutExpired:
            report = {"report_version": VERSION, "status": "partial", "file": {"name": filename, "bytes": size, "sha256": "не вычислен"},
                      "error": {"type": "WorkerTimeout", "message": "Превышен лимит {} секунд.".format(self.server.worker_timeout)},
                      "findings_ru": ["Процесс разбора остановлен по тайм-ауту. Передайте частичный отчёт."],
                      "warnings_ru": ["Модель не запускалась."]}
            self.finish_report(run_id, report)
        except (BrokenPipeError, ConnectionResetError):
            print("[{}] Браузер отключился. Проверьте папку reports.".format(run_id), flush=True)
        except Exception as exc:
            print("[{}] {}: {}".format(run_id, type(exc).__name__, str(exc)[:500]), flush=True)
            self.send(400, {"error": "{}: {}".format(type(exc).__name__, str(exc)[:1000])})
        finally:
            self.server.slot.release()

    def finish_report(self, run_id, report):
        report["run_id"] = run_id
        rendered = markdown_report(report)
        prefix = self.server.report_dir / run_id
        prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        prefix.with_suffix(".md").write_text(rendered, encoding="utf-8")
        print("[{}] {}. Отчёты: {}.md и .json".format(run_id, report["status"], prefix.resolve()), flush=True)
        self.send(200, {"report": report, "markdown": rendered})


def inspect_failure(path, filename, message):
    return {"report_version": VERSION, "status": "partial",
            "file": {"name": filename, "bytes": Path(path).stat().st_size, "sha256": "не вычислен"},
            "error": {"type": "WorkerFailure", "message": message},
            "findings_ru": ["Разбор завершился ошибкой. Это диагностический отчёт, не результат предсказания."],
            "warnings_ru": ["Веса не запускались; запуск нейросети этим сервером не поддерживается."]}


def main():
    parser = argparse.ArgumentParser(description="Локальная диагностика .pt без PyTorch. Python 3.8+; устанавливать библиотеки не нужно.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--max-upload-mb", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--worker", nargs=2, metavar=("INPUT", "OUTPUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        report = inspect_file(args.worker[0])
        Path(args.worker[1]).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        return
    if args.max_upload_mb < 1 or not 1 <= args.timeout_seconds <= 120:
        parser.error("Размер должен быть положительным, тайм-аут — от 1 до 120 секунд.")
    print("Проверка зависимостей: Python {}. Все библиотеки встроенные; pip install не нужен.".format(sys.version.split()[0]), flush=True)
    try:
        server = DiagnosticServer((args.host, args.port), args.reports_dir, args.max_upload_mb * 1024**2, args.timeout_seconds)
    except OSError as exc:
        raise SystemExit("Не удалось запустить сервер: {}. Если порт занят, добавьте --port 8766.".format(exc))
    host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print("\nОткройте ВЕСЬ адрес в браузере на этом ПК:\nhttp://{}:{}/#token={}\n".format(host, server.server_port, server.token), flush=True)
    print("Отчёты: {}\nЛимит файла: {} МБ. Тайм-аут разбора: {} сек. Остановка: Ctrl+C.".format(
        server.report_dir.resolve(), args.max_upload_mb, args.timeout_seconds), flush=True)
    if args.host not in {"127.0.0.1", "localhost"}:
        print("ВНИМАНИЕ: доступ открыт по сети. Только доверенная локальная сеть; HTTP без шифрования. Не публикуйте в интернете.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
