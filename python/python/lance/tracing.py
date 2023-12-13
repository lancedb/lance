#  Copyright (c) 2023. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import atexit

from .lance import trace_to_chrome as lance_trace_to_chrome


def trace_to_chrome(*, file: str = None, level: str = None):
    """
    Begins tracing lance events to a chrome trace file.

    The trace file can be opened with chrome://tracing or with the Perfetto UI.

    The file will be finished (and closed) when the python process exits.

    Parameters
    ----------
    file: str, optional
        The file to write the trace to. If None, then a file in the current
        directory will be created named ./trace-{unix epoch in micros}.json
    level: str, optional
        The level of detail to trace. One of "trace", "debug", "info", "warn"
        or "error".  If None, then "info" is used.
    """
    guard = lance_trace_to_chrome(file, level)
    atexit.register(lambda: guard.finish_tracing())
