from __future__ import annotations

import os
import tempfile
from pathlib import Path


TEST_ROOT = Path(tempfile.mkdtemp(prefix="cashgap-tests-"))
os.environ["CASHGAP_ROOT"] = str(TEST_ROOT)
os.environ["CASHGAP_DB_PATH"] = str(TEST_ROOT / "cashgap-test.db")

