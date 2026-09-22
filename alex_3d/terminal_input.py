"""Non-blocking single-key commands read from the launching terminal."""

from __future__ import annotations

import os
import select
import sys


class TerminalKeyReader:
    """Read terminal keys without requiring Enter when stdin is a TTY."""

    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self._attributes = None

    def __enter__(self):
        if self.stream.isatty():
            import termios
            import tty

            descriptor = self.stream.fileno()
            self._attributes = termios.tcgetattr(descriptor)
            tty.setcbreak(descriptor)
            attributes = termios.tcgetattr(descriptor)
            attributes[3] &= ~termios.ECHO
            termios.tcsetattr(descriptor, termios.TCSADRAIN, attributes)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._attributes is not None:
            import termios

            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self._attributes)
            self._attributes = None

    def poll(self) -> str | None:
        readable, _, _ = select.select([self.stream], [], [], 0)
        if not readable:
            return None
        character = os.read(self.stream.fileno(), 1).decode(errors="ignore")
        if character in ("\r", "\n"):
            return "enter"
        return character.lower() if character else None
