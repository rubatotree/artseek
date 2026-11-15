from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from artseek.server_output import server_output


class ServerOutputTests(unittest.TestCase):
    def test_closed_launcher_pipes_do_not_break_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            code = """
import sys
from pathlib import Path
original_stderr = sys.stderr
from artseek.server_output import server_output
with server_output(Path(sys.argv[1])):
    print('index progress: 1/1', flush=True)
    print('cached handler warning', file=original_stderr, flush=True)
    print('model warning preserved', file=sys.stderr, flush=True)
"""
            # No reader survives the launch, just as when a GUI supervisor exits.
            read_fd, write_fd = os.pipe()
            os.close(read_fd)
            try:
                result = subprocess.run(
                    [sys.executable, "-c", code, directory],
                    stdout=write_fd,
                    stderr=write_fd,
                    timeout=30,
                    check=False,
                )
            finally:
                os.close(write_fd)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(
                (base / "logs" / "server.log").read_text(encoding="utf-8"),
                "index progress: 1/1\ncached handler warning\nmodel warning preserved\n",
            )

    def test_explicit_output_file_is_preserved_and_work_errors_propagate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            destination = base / "chosen.log"
            with destination.open("w", encoding="utf-8") as output:
                with redirect_stdout(output), redirect_stderr(output):
                    with self.assertRaisesRegex(OSError, "image read failed"):
                        with server_output(base):
                            print("progress retained")
                            raise OSError("image read failed")
                    print("streams restored")
            self.assertEqual(
                destination.read_text(encoding="utf-8"),
                "progress retained\nstreams restored\n",
            )
            self.assertFalse((base / "logs").exists())


if __name__ == "__main__":
    unittest.main()
