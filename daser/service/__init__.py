# SPDX-License-Identifier: Apache-2.0
"""HTTP service layer for DaseR.

Exposes a FastAPI-based REST API for uploading documents, listing the
document registry and running inference against selected documents.
All state is delegated to the DaseR control plane via the Unix-socket
IPC protocol — this module is deliberately stateless.
"""
