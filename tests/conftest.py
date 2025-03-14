# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2 import setup_fairseq2
from pytest import Session


def pytest_sessionstart(session: Session) -> None:
    setup_fairseq2()
