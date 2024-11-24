# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import logging

from ci_benchmarks.datagen.lineitems import gen_tcph

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen_tcph()
