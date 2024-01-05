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

from typing import Optional

from lance.dependencies import torch


def preferred_device(device: Optional[str] = None):
    """Get the preferred device for computation.

    Parameters
    ----------
    device : str, optional
        Device to use for computation. If None, the device will be
        detected automatically based on the platform.

    Returns
    -------
    device : torch.device
        Device to use for computation.
    """
    if device is not None:
        if isinstance(device, str):
            device = torch.device(device)
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")
