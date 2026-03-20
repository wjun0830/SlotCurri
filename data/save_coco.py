"""Script to download and shard COCO 2017."""
import argparse
import os
import zipfile

import cv2
import numpy as np
import tqdm
import webdataset as wds
from PIL import Image
from pycocotools.coco import COCO
from utils import download_file, get_default_dir

TARGET_SIZE = 256

IMAGE_URL = "http://images.cocodataset.org/zips"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations"
SPLITS_TO_SUFFIX = {
    "train": "train2017",
    "validation": "val2017",
    "test": "test2017",
    "unlabeled": "unlabeled2017",
}
SPLITS_TO_ANNOTATIONS_ZIP = {
    "train": "annotations_trainval2017.zip",
    "validation": "annotations_trainval2017.zip",
    "test": "image_info_test2017.zip",
    "unlabeled": "image_info_unlabeled2017.zip",
}
SPLITS_TO_ANNOTATIONS_FILENAME = {
    "train": "instances_train2017.json",
    "validation": "instances_val2017.json",
    "test": "image_info_test2017.json",
    "unlabeled": "image_info_unlabeled2017.json",
}


parser = argparse.ArgumentParser("Generate sharded dataset from original COCO data.")
parser.add_argument(
    "--split", default="train", choices=list(SPLITS_TO_SUFFIX), help="Which splits to write"
)
parser.add_argument(
    "--download-dir",
    default=get_default_dir("coco_raw"),
    help="Directory where COCO images and annotations are downloaded to",
)
parser.add_argument("--maxcount", type=int, default=256, help="Max number of samples per shard")
parser.add_argument("--only-images", action="store_true", help="Whether to store only images")
parser.add_argument(
    "--original-image-format",
    action="store_true",
    help="Whether to keep the orginal image encoding (e.g. jpeg), or convert to numpy",
)
parser.add_argument(
    "--out-path",
    default=get_default_dir("coco256"),
    help="Directory where shards are written",
)


def download_zip_and_extract(url, dest_dir):
    print(f"Downloading {url} to {dest_dir}")
    file = download_file(url, dest_dir)
    print(f"\nExtracting {file} to {dest_dir}")
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
    os.remove(file)


def get_coco_images(data_dir, split):
    assert split in SPLITS_TO_SUFFIX
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    download_zip_and_extract(f"{IMAGE_URL}/{SPLITS_TO_SUFFIX[split]}.zip", image_dir)
    return image_dir


def get_coco_annotations(data_dir, annotation_zip):
    os.makedirs(data_dir, exist_ok=True)
    download_zip_and_extract(f"{ANNOTATIONS_URL}/{annotation_zip}", data_dir)
    return data_dir


def resize_image(arr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(arr, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.uint8)


def resize_masks(masks: np.ndarray) -> np.ndarray:
    num_obj, h, w = masks.shape
    resized = np.zeros((num_obj, TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    for idx in range(num_obj):
        resized[idx] = cv2.resize(
            masks[idx], (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST
        )
    return resized


def write_dataset(
    image_dir,
    out_dir,
    split,
    add_annotations=True,
    annotations_file=None,
    convert_images_to_numpy=True,
):
    if add_annotations and annotations_file is None:
        raise ValueError("Requested to add annotations but annotations file not given")

    if not os.path.isdir(os.path.join(out_dir, ".")):
        os.makedirs(os.path.join(out_dir, "."), exist_ok=True)

    # This is the output pattern under which we write shards.
    pattern = os.path.join(out_dir, f"coco-{split}-%06d.tar")

    coco = COCO(annotations_file)
    with wds.ShardWriter(pattern, maxcount=args.maxcount) as sink:
        max_num_instances = 0
        for img_id, img in tqdm.tqdm(coco.imgs.items()):
            img_filename = coco.loadImgs(img["id"])[0]["file_name"]
            img_path = os.path.join(image_dir, img_filename)

            with Image.open(img_path).convert("RGB") as img_obj:
                img_obj.load()
                image_arr = np.asarray(img_obj, dtype=np.uint8)
            image_arr = resize_image(image_arr)

            if not add_annotations:
                sample = {"__key__": str(img_id)}
            else:
                ann_ids = coco.getAnnIds(img_id, iscrowd=False)  # We filter out crowd labels
                anns = coco.loadAnns(ann_ids)
                if anns:
                    segmentations = np.stack(
                        [coco.annToMask(ann) * ann["category_id"] for ann in anns]
                    ).astype(np.uint8)
                    assert len(segmentations.shape) == 3
                    max_num_instances = max(max_num_instances, len(anns))
                else:
                    h, w = image_arr.shape[:2]
                    segmentations = np.zeros(shape=(1, h, w), dtype=np.uint8)
                segmentations = resize_masks(segmentations)
                sample = {
                    "__key__": str(img_id),
                    "segmentations.npy": segmentations,
                }
            if convert_images_to_numpy:
                sample["image.npy"] = image_arr
            else:
                ext = os.path.splitext(img_filename)[-1] or ".jpg"
                bgr = cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(ext, bgr)
                if not ok:
                    raise RuntimeError("Failed to encode resized image")
                sample[f"image{ext}"] = buf.tobytes()
            sink.write(sample)

        if add_annotations:
            print(f"Maximal number of instances per image in {split} split is: {max_num_instances}.")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.split in ("test", "unlabeled") and not args.only_images:
        raise NotImplementedError(
            f"Split {args.split} does not have any annotations. Run with --only-images"
        )

    if args.split in SPLITS_TO_ANNOTATIONS_FILENAME:
        annotations_dir = os.path.join(args.download_dir, "annotations")
        annotations_file = os.path.join(annotations_dir, SPLITS_TO_ANNOTATIONS_FILENAME[args.split])
        if not os.path.exists(annotations_file):
            zip_file = SPLITS_TO_ANNOTATIONS_ZIP[args.split]
            get_coco_annotations(args.download_dir, zip_file)
        assert os.path.exists(annotations_file)
    else:
        annotations_file = None

    image_dir = os.path.join(args.download_dir, "images", SPLITS_TO_SUFFIX[args.split])
    if not os.path.exists(image_dir):
        get_coco_images(args.download_dir, args.split)
        assert os.path.exists(image_dir)

    write_dataset(
        image_dir=image_dir,
        out_dir=args.out_path,
        split=args.split,
        add_annotations=annotations_file is not None and not args.only_images,
        annotations_file=annotations_file,
        convert_images_to_numpy=not args.original_image_format,
    )
