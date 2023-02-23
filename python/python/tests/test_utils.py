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

import os
from pathlib import Path

from lance.util import sanitize_file_address


def test_sanitize_path():
    # string addresses
    addr = "~/foo.lance"
    sanitized_addr = sanitize_file_address(addr)
    assert len(sanitized_addr) > len(addr)  # from tilde expansion

    for addr in ["/bar/foo.lance", "s3://bucket/foo.lance", "../foo.lance"]:
        sanitized_addr = sanitize_file_address(addr)
        assert sanitized_addr == addr  # no expansion needed here

    # Path objects
    addr = Path("/bar/foo.lance")
    sanitized_addr = sanitize_file_address(addr)
    assert sanitized_addr == os.fspath(addr)  # no expansion needed here

    addr = Path("~/foo.lance")
    sanitized_addr = sanitize_file_address(addr)
    assert len(sanitized_addr) > len(os.fspath(addr))  # from tilde expansion

    for addr in [Path("foo.lance"), Path("../bar/foo.lance")]:
        sanitized_addr = sanitize_file_address(addr)
        assert len(sanitized_addr) > len(os.fspath(addr))  # absolute address expansion
