# Copyright (c) 2026 Dedalus Labs, Inc. and its contributors
# SPDX-License-Identifier: MIT

"""Entrypoint."""

import asyncio

from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    from server import main

    asyncio.run(main())
