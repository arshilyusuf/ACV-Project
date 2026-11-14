# Dataset card

## Dataset
- **Name/source:** Oxford-IIIT Pet Dataset, official files in `data/images` and `data/annotations`.
- **Publication:** O. M. Parkhi et al., *Cats and Dogs*, CVPR 2012.
- **License/terms:** Research-use dataset; images retain the terms of their original source websites (see `data/annotations/README`).
- **Images:** 7390 total images; 7349 images are covered by the official trainval/test lists (3312 train, 368 validation, 3669 test).

## Data contract
- **Original masks:** trimap values 1=foreground, 2=background, 3=not classified/boundary.
- **Remapped masks:** class 0=background, class 1=foreground, value 255=ignore.
- **Ignore policy:** 255 is ignored by `CrossEntropyLoss`, confusion matrices, IoU, and Dice; it is not a class.
- **Image preprocessing:** RGB, resize to 128x128 with bilinear interpolation, scale to [0,1], ImageNet mean/std normalization.
- **Mask preprocessing:** grayscale, resize to 128x128 with nearest-neighbor interpolation, then remap; nearest-neighbor prevents invalid labels.
- **Training augmentation:** random horizontal flip (p=0.5). Validation/test: resize and normalization only.
- **Split:** seed `20261114`; `random.Random(seed)` shuffles official `trainval.txt`, first 90% is train and remaining 10% validation; official `test.txt` is test. Exact IDs are in `outputs/splits/dataset_splits.json`.
- **Ignored-pixel percentage:** 11.5782% over all paired images.
