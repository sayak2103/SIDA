
#!/bin/bash
# Run the training script with DeepSpeed directly specifying GPU 1
deepspeed --include localhost:1 --master_port=24999 train_SIDA.py \
  --version="./ck/LISA-7B-v1" \
  --dataset_dir='/LOCAL2/zhenglin/benchmark' \
  --vision_pretrained="/LOCAL2/zhenglin/LISA/ck/sam_vit_h_4b8939.pth" \
  --val_dataset="/LOCAL2/zhenglin/benchmark/"\
  --batch_size=2 \
  --exp_name="SIDA-7B-test" \
  --epochs=10 \
  --steps_per_epoch=1000 \
  --lr=0.0001 \
