#  Copyright (c) 2022. Lance Developers
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

import base64
from pathlib import Path

import requests
import requests_mock

from lance.io import open_uri

WIKIPEDIA = (
    "https://upload.wikimedia.org/wikipedia/commons/a/ad/"
    "Commodore_Grace_M._Hopper%2C_USN_%28covered%29.jpg"
)


def test_open_https_uri():
    with open_uri(WIKIPEDIA) as fobj:
        assert len(fobj.read()) > 0


def test_local(tmp_path: Path):
    with open_uri(WIKIPEDIA) as fobj:
        img_bytes = fobj.read()
        with open_uri(tmp_path / "wikipedia.jpg", mode="wb") as fobj:
            fobj.write(img_bytes)
        with open_uri(tmp_path / "wikipedia.jpg") as fobj:
            assert img_bytes == fobj.read()


def test_simple_http_credentials():
    with requests_mock.Mocker() as mock:
        mock.get("http://test.com", text="{}")
        requests.get("http://test.com", auth=("user", "def_not_pass"))
        req = mock.request_history[0]
        assert req.headers.get("Authorization") == "Basic {}".format(
            base64.b64encode(b"user:def_not_pass").decode("utf-8")
        )


def test_no_http_credentials():
    with requests_mock.Mocker() as mock:
        mock.get("http://test.com", text="{}")
        requests.get("http://test.com")
        req = mock.request_history[0]
        assert "Authorization" not in req.headers
