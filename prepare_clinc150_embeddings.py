from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
from torch.nn import functional as F


def hash_embedding(text: str, dimension: int) -> torch.Tensor:
    """Deterministic smoke-test embedding; not valid for paper results."""
    result = torch.zeros(dimension)
    for token in re.findall(r"[a-z0-9']+", text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:8], "little") % dimension
        result[index] += -1.0 if digest[8] & 1 else 1.0
    return F.normalize(result, dim=0) if result.norm() else result


def main(args):
    raw = json.loads(Path(args.data).read_text())
    rows = []
    texts = []
    for split in ("train", "val", "test", "oos_train", "oos_val", "oos_test"):
        for text, intent in raw[split]:
            rows.append({"index": len(rows), "split": split.replace("val", "validation"), "intent": intent})
            texts.append(text)
    if args.encoder == "hash":
        embeddings = torch.stack([hash_embedding(text, args.dimension) for text in texts])
        encoder_name = f"hash-smoke-{args.dimension}"
    else:
        try:



            project_root = str(Path(__file__).resolve().parent)
            original_path = list(sys.path)
            sys.path = [entry for entry in sys.path if entry not in ("", project_root)]
            sys.modules.pop("datasets", None)
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise SystemExit(f"Cannot import sentence-transformers: {error}") from error
        finally:
            sys.path = original_path
        model = SentenceTransformer(args.encoder)
        embeddings = torch.as_tensor(model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True))
        encoder_name = args.encoder
    intents = sorted({row["intent"] for row in rows if row["intent"] != "oos"})
    payload = {
        "embeddings": embeddings.float(),
        "rows": rows,
        "intent_to_id": {name: index for index, name in enumerate(intents)},
        "encoder": encoder_name,
        "paper_valid": args.encoder != "hash",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": args.output, "shape": list(embeddings.shape), "paper_valid": payload["paper_valid"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="vendor/oos-eval/data/data_full.json")
    parser.add_argument("--output", default="clinc150_features/clinc150_embeddings.pt")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=128)
    main(parser.parse_args())
