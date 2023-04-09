#names=(test_meanstd_00_01 test_meanstd_05_05 test_minmax)
names=(test_meanstd_05_05)
anns=(ann_dir cross_dir)
ths=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for name in $names
do
  for ann in $anns
  do
    for th in $ths
    do
      python tools/diffusemade/gen_label_and_prob.py --test --root data/DiffuseMade_test5 \
        --img-dir "${name}/img_dir/train" \
        --ann-dir "${name}/${ann}/train" \
        --out-dir "output/${name}-${ann}" \
        --refer deeplabv3_pseudo --refrain \
        --low ${th} --high ${th} \
        --post no \
        --eval-only \
        --pre-act no \
        --eval mIoU mFscore;
    done
  done
done