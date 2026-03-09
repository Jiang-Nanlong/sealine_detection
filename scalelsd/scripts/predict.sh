
test_root=data-ssl/hybrid_dataset/COCO/val2017

threshold=10
jhm=0.1
res=512

python -m predictor.predict \
    --img $test_root \
    --ext png \
    --threshold $threshold \
    --width $res --height $res  \
    --junction-hm $jhm \
    --disable-show  \
    # --use_lsd \
    # --use_nms \
    # --whitebg 1.
    