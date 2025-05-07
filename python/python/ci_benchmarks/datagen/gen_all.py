# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from ci_benchmarks.datagen.basic import gen_basic
from ci_benchmarks.datagen.lineitems import gen_tcph

if __name__ == "__main__":
    gen_basic()
    gen_tcph()
