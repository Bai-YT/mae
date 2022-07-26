## Pre-training MAE

To pre-train with **multi-node distributed training**, run the following on 8 nodes with 8 GPUs each:

Use the following for Submitit deployment (not tested):
```
python3 submitit_pretrain.py \
    --job_dir "/home/ubuntu/project/MAE/LGS+IN/" \
    --nodes 1 \
    --use_volta32 \
    --batch_size 64 \
    --accum_iter 4 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path "/home/ubuntu/project/classification/data/IN_data; /home/ubuntu/project/LGS_710/images"
```

Use the following for deployment WITHOUT Submitit (tested):
```
python3.10 -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=100.64.214.173 \
    --master_port=8840 \
    main_pretrain.py \
    --batch_size 85 \
    --accum_iter 3 \
    --num_workers 2 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.75 \
    --epochs 400 \
    --warmup_epochs 20 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --norm_pix_loss \
    --pin_mem \
    --data_path "/home/ubuntu/project/data/IN_data; /home/ubuntu/project/data/LGS" \
    --use_lgs \
    --log_dir "/home/ubuntu/project/MAE/LGS+IN/norm_loss/" \
    --output_dir "/home/ubuntu/project/MAE/LGS+IN/norm_loss/"
```

ImageNet-only baseline:
```
python3.10 -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=100.64.181.53 \
    --master_port=8870 \
    main_pretrain.py \
    --batch_size 128 \
    --accum_iter 2 \
    --num_workers 4 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --pin_mem \
    --norm_pix_loss \
    --data_path "/home/ubuntu/project/data/IN_data; /home/ubuntu/project/data/LGS" \
    --log_dir "/home/ubuntu/project/MAE/IN/norm_loss/" \
    --output_dir "/home/ubuntu/project/MAE/IN/norm_loss/"
```

Re-pretrain based on the official ImageNet model (no norm pix loss):
```
python3 -m torch.distributed.launch  --nproc_per_node=8  --nnodes=1  --node_rank=0  --master_addr=100.64.214.173  --master_port=8840  main_pretrain.py  --batch_size 85  --accum_iter 3  --num_workers 2  --model mae_vit_base_patch16  --mask_ratio 0.75  --epochs 150  --warmup_epochs 20  --blr 1e-4  --weight_decay 0.05  --pin_mem  --data_path "/home/ubuntu/project/data/IN_data; /home/ubuntu/project/data/LGS"  --use_lgs  --log_dir "/home/ubuntu/project/MAE/LGS+IN/nonorm_loss_pretrained/"  --output_dir "/home/ubuntu/project/MAE/LGS+IN/nonorm_loss_pretrained/"  --resume "/home/ubuntu/project/MAE/IN/official/mae_visualize_vit_base.pth"
```
With balanced LGS:
```
python3 -m torch.distributed.launch  --nproc_per_node=8  --nnodes=1  --node_rank=0  --master_addr=100.64.229.103  --master_port=4000  main_pretrain.py  --batch_size 64  --accum_iter 4  --num_workers 4  --model mae_vit_base_patch16  --mask_ratio 0.75  --epochs 150  --warmup_epochs 20  --blr 1e-4  --weight_decay 0.05  --pin_mem  --data_path "/home/ubuntu/project/data/IN_data; /home/ubuntu/project/data/LGS"  --use_lgs  --weighted_lgs  --log_dir "/home/ubuntu/project/MAE/LGS+IN/nonorm_loss_pretrained_balanced/"  --output_dir "/home/ubuntu/project/MAE/LGS+IN/nonorm_loss_pretrained_balanced/"  --resume "/home/ubuntu/project/MAE/LGS+IN/nonorm_loss_pretrained_balanced/checkpoint-1.pth"  --start_epoch 2
```

Use the following for debugging (no DDP):
```
python3 \
    main_pretrain.py \
    --batch_size 64 \
    --epochs 800 \
    --accum_iter 4 \
    --model mae_vit_base_patch16 \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --weight_decay 0.05 \
    --blr 1.5e-4 \
    --warmup_epochs 40 \
    --data_path '/home/ubuntu/project/classification/data/IN_data; /home/ubuntu/project/LGS_710/images' \
    --log_dir '/home/ubuntu/project/MAE/LGS+IN/' \
    --output_dir '/home/ubuntu/project/MAE/LGS+IN/' \
    --num_workers 20 \
    --pin_mem
```

- Here the effective batch size is 64 (`batch_size` per gpu) * 8 (`nodes`) * 8 (gpus per node) = 4096. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is `batch_size` (per gpu) * `nodes` * 8 (gpus per node) * `accum_iter`.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.
- Here we use `--norm_pix_loss` as the target for better representation learning. To train a baseline model (e.g., for visualization), use pixel-based construction and turn off `--norm_pix_loss`.
- The exact same hyper-parameters and configs (initialization, augmentation, etc.) are used as our TF/TPU implementation. In our sanity checks, this PT/GPU re-implementation can reproduce the TF/TPU results within reasonable random variation. We get 85.5% [fine-tuning](FINETUNE.md) accuracy by pre-training ViT-Large for 800 epochs (85.4% in paper Table 1d with TF/TPU).
- Training time is ~42h in 64 V100 GPUs (800 epochs).

To train ViT-Base or ViT-Huge, set `--model mae_vit_base_patch16` or `--model mae_vit_huge_patch14`.
