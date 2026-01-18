from dataset import CraterDataset

ds = CraterDataset(
    gt_csv="../nasa-craters-data/train-gt.csv",
    cv_csv="../crater_baseline/output/solution.csv",
    img_root="../nasa-craters-data/train"
)

print(len(ds))
img, target = ds[0]
print(img.shape, target)
