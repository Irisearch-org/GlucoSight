"""Reconcile food-10 MobileNetV2 test accuracy across code paths.

Usage (from the repo root; weights and dataset live outside git):
    python cv/scripts/eval_food10_reconcile.py \
        --checkpoint /abs/path/cv/models/mobilenetv2_food10.pt \
        --test-root "/abs/path/Indonesian Food Image/Clean_Data/test" \
        --out-json /tmp/food10_reconcile.json \
        --out-predictions /tmp/food10_predictions.jsonl

Paths measured:
  app   : cv_baseline.preprocessing.preprocess + MobileNetV2Classifier (what predict() runs)
  train : torchvision Resize(255)+CenterCrop(224)+ImageNet norm (train_mobilenet.py / notebook eval_tf)

Also reconstructs the 99-image VLM comparison sample (run_gas_vlm.gather_test_images:
random.seed(42) over the sorted test list, minus pic_338) and scores it by path and by
bare filename stem (the key used by evaluate_real_vs_gt*.py and run_gas_vlm.py).
Stem-keyed scoring is ambiguous when one stem exists in several class folders, so
the script reports a lower and upper bound over all possible stem resolutions.
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "cv"))

import torch  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from torchvision import models, transforms  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

from cv_baseline.classifier import MobileNetV2Classifier  # noqa: E402
from cv_baseline.preprocessing import preprocess  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def gather(test_root: Path):
    """Same ordering as run_gas_vlm.gather_test_images (before sampling)."""
    items = []
    for cls_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in IMG_EXT:
                items.append((img.stem, str(img), cls_dir.name))
    return items


def build_torch_model(ckpt):
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, ckpt["num_classes"])
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def metrics(y_true, top3, classes):
    n = len(y_true)
    top1 = [k[0] for k in top3]
    t1 = sum(int(p == t) for t, p in zip(y_true, top1))
    t3 = sum(int(t in k) for t, k in zip(y_true, top3))
    y_idx = [classes.index(t) for t in y_true]
    p_idx = [classes.index(p) for p in top1]
    macro_f1 = f1_score(y_idx, p_idx, labels=list(range(len(classes))), average="macro", zero_division=0)
    per = {}
    for c in classes:
        idx = [i for i, t in enumerate(y_true) if t == c]
        per[c] = round(sum(top1[i] == c for i in idx) / len(idx), 4) if idx else None
    return {"n": n, "top1_correct": t1, "top1": round(t1 / n, 4),
            "top3_correct": t3, "top3": round(t3 / n, 4),
            "macro_f1": round(float(macro_f1), 4), "per_class_top1": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--test-root", required=True, type=Path)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-predictions", type=Path)
    args = ap.parse_args()

    items = gather(args.test_root)
    classes = sorted({c for _, _, c in items})
    print(f"test images: {len(items)} across {len(classes)} classes", file=sys.stderr)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_to_idx = ckpt["class_to_idx"]
    assert [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])] == classes, \
        "checkpoint class order != test folder order"

    # --- app path -------------------------------------------------------
    clf = MobileNetV2Classifier(model_path=args.checkpoint).load()
    app_top = {path: [c for c, _ in clf.predict_topk(preprocess(path), k=3)] for _, path, _ in items}

    # --- training path --------------------------------------------------
    tf = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    ds = ImageFolder(str(args.test_root), transform=tf)
    model = build_torch_model(ckpt)
    train_top = {}
    with torch.no_grad():
        for i, (x, _) in enumerate(ds):
            order = torch.argsort(model(x.unsqueeze(0))[0], descending=True)[:3].tolist()
            train_top[ds.samples[i][0]] = [classes[j] for j in order]
    assert len(train_top) == len(items)

    y = [c for _, _, c in items]
    app_pred = [app_top[p] for _, p, _ in items]
    train_pred = [train_top[p] for _, p, _ in items]
    res_app = metrics(y, app_pred, classes)
    res_train = metrics(y, train_pred, classes)
    agree = sum(a[0] == b[0] for a, b in zip(app_pred, train_pred))

    # --- 99-image VLM sample ---------------------------------------------
    random.seed(42)
    sample = random.sample(items, min(100, len(items)))
    sample99 = [s for s in sample if s[0] != "pic_338"]
    path_correct = 0
    stem_classes = defaultdict(set)
    for sid, path, cls in items:
        stem_classes[sid].add(cls)
    lower = upper = 0
    for sid, path, cls in sample99:
        pred = app_top[path][0]
        path_correct += int(pred == cls)
        if len(stem_classes[sid]) == 1:
            lower += int(pred == cls)
            upper += int(pred == cls)
        else:
            # any class in the stem's set may be the one the dict keeps
            upper += int(pred in stem_classes[sid])
    ambiguous99 = sum(len(stem_classes[sid]) > 1 for sid, _, _ in sample99)

    stems = Counter(sid for sid, _, _ in items)
    out = {
        "checkpoint": str(args.checkpoint),
        "app_path_full_test": res_app,
        "train_path_full_test": res_train,
        "app_vs_train_top1_agreement": round(agree / len(items), 4),
        "stem_collisions": {"stems_shared_by_>1_class": sum(v > 1 for v in stems.values()),
                            "images_with_shared_stem": sum(v for v in stems.values() if v > 1)},
        "vlm_sample_99": {
            "n": len(sample99),
            "excluded": "pic_338",
            "app_top1_path_keyed": f"{path_correct}/{len(sample99)}",
            "app_top1_stem_keyed_bounds": f"{lower}/{len(sample99)} .. {upper}/{len(sample99)}",
            "images_whose_stem_maps_to_multiple_classes": ambiguous99,
        },
    }
    print(json.dumps(out, indent=2))
    if args.out_json:
        args.out_json.write_text(json.dumps(out, indent=2))
    if args.out_predictions:
        with args.out_predictions.open("w") as f:
            for sid, path, cls in items:
                f.write(json.dumps({"sample_id": sid, "path": path, "true": cls,
                                    "app_top3": app_top[path], "train_top3": train_top[path]}) + "\n")


if __name__ == "__main__":
    main()
