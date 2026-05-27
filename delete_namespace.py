"""
delete_namespace.py
-------------------
One-time utility to delete a specific PDF's namespace from Pinecone,
forcing it to be re-indexed with the current loader on next upload.

Usage
-----
    # List all namespaces first to find the right one:
    python delete_namespace.py --list

    # Delete namespace for a specific PDF file:
    python delete_namespace.py --pdf attention_is_all_you_need.pdf

    # Delete ALL namespaces (full wipe):
    python delete_namespace.py --all

    # Delete by namespace hash directly:
    python delete_namespace.py --namespace abc12345
"""

import argparse
import hashlib
import time

from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
from config import cfg


def _get_index():
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    return pc.Index(cfg.pinecone_index)


def get_doc_hash(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def list_namespaces():
    index = _get_index()
    stats = index.describe_index_stats()
    namespaces = stats.namespaces
    if not namespaces:
        print("No namespaces found.")
        return
    print(f"Namespaces in index '{cfg.pinecone_index}':")
    for ns, summary in namespaces.items():
        display = repr(ns) if not ns else ns
        print(f"  {display:<20} {summary.vector_count} vectors")


def delete_namespace(namespace: str):
    index = _get_index()
    stats = index.describe_index_stats()

    if namespace not in stats.namespaces:
        print(f"Namespace '{namespace}' not found — nothing to delete.")
        return

    count = stats.namespaces[namespace].vector_count
    index.delete(delete_all=True, namespace=namespace)

    # Confirm deletion by polling
    print(f"Deleting namespace '{namespace}' ({count} vectors)...", end="", flush=True)
    for _ in range(10):
        time.sleep(2)
        stats = index.describe_index_stats()
        if namespace not in stats.namespaces:
            print(" ✓ Confirmed deleted.")
            print("Re-upload the PDF in the app to re-index it cleanly.")
            return
        print(".", end="", flush=True)

    print("\nWarning: namespace still appears in stats — Pinecone may need more time.")
    print("Wait 30 seconds and run --list again to confirm before re-uploading.")


def delete_all():
    index = _get_index()
    stats = index.describe_index_stats()
    namespaces = list(stats.namespaces.keys())
    if not namespaces:
        print("No namespaces to delete.")
        return
    for ns in namespaces:
        display = repr(ns) if not ns else ns
        print(f"Deleting namespace: {display}")
        index.delete(delete_all=True, namespace=ns)
    print(f"Deleted {len(namespaces)} namespace(s). Re-upload your PDF to re-index.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Path to PDF whose namespace should be deleted.")
    group.add_argument("--namespace", help="Pinecone namespace hash to delete directly.")
    group.add_argument("--all", action="store_true", help="Delete ALL namespaces.")
    group.add_argument("--list", action="store_true", help="List all namespaces.")
    args = parser.parse_args()

    if args.list:
        list_namespaces()
    elif args.all:
        delete_all()
    elif args.pdf:
        ns = get_doc_hash(args.pdf)
        print(f"Hash for '{args.pdf}': {ns}")
        delete_namespace(ns)
    else:
        delete_namespace(args.namespace)
