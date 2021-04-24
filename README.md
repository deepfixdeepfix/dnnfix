## reproduce results of repairing confusion on cifar10

#### step1: clone cutmix repository
```
git clone https://github.com/clovaai/CutMix-PyTorch.git
```

#### step2: install the necessary environment from https://github.com/clovaai/CutMix-PyTorch
#### see https://github.com/clovaai/CutMix-PyTorch

#### step3: copy all files in dnnfix/cifar10 to cutmix folder and train baseline model
```
python3 train_baseline.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname cifar10_resnet18_2_4 --epochs 300 --beta 1.0 --cutmix_prob 0
 
```
#### step4: check model cat-dog confusion
```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel --first 3 --second 5
```
#### step4: check model cat-dog confusion

##### [w-aug]:  

```
python3 repair_confusion_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --weight 3.0
```

##### [w-bn]:  

```
python3 repair_confusion_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --first 3 --second 5 --ratio 0.2  
```

##### [w-os]:
```
python3 repair_confusion_exp_newbn_softmax.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.1 --checkmodel --first 3 --second 5
```

##### [w-loss]:
```
python3 repair_confusion_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --target_weight 0.4
```
##### [w-dbr]:  

```
python3 repair_confusion_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --first 3 --second 5 --lam 0.5
```


## reproduce results if repairing bias on cifar10
### set up environment same as above
#### [orig]:    

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --checkmodel  
```

#### [w-aug]:  

```
python3 repair_bias_exp_oversampling.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --third 2 --weight 2.0
```

#### [w-bn]:  

```
python3 repair_bias_exp_newbn.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --replace --ratio 0.2 --first 3 --second 5 --third 2
```

#### [w-os]:
```
python3 repair_bias_exp_newbn_softmax.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --eta 0.3 --checkmodel --first 3 --second 5 --third 2
```

#### [w-loss]:
```
python3 repair_bias_exp_weighted_loss.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 128 --lr 0.1 --expname cifar10_resnet_2_4_dogcat_test --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --lam 0 --extra 128 --first 3 --second 5 --third 2 --target_weight 0.4
```
#### [w-dbr]:  

```
python3 repair_bias_dbr.py --net_type resnet --dataset cifar10 --depth 18 --batch_size 256 --lr 0.1 --expname ResNet18 --epochs 60 --beta 1.0 --cutmix_prob 0 --pretrained ./runs/cifar10_resnet18_2_4/model_best.pth.tar --expid 0 --first 3 --second 5 --third 2 --lam 0.5
```



## reproduce results of repairing confusion on coco

### Train original model
```
python2 train_epoch_graph.py --log_dir original_model --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/'
```

#### [w-aug]:  

```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair_aug --first "person" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 3 --class_num 80
```

#### [w-bn]:  

```
python2 repair_confusion_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair_bn --first "person" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.4
```

#### [w-os]:
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode confusion --first "bus" --second "person"
```

#### [w-loss]:
```
python2 repair_confusion_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair_loss --first "bus" --second "person" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 80 --target_weight 0.4
```
#### [w-dbr]:  

```
python2 repair_confusion_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_confusion_repair_dbr --first "person" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --lam 0.5
```

## reproduce results of repairing bias on coco

#### [w-aug]:  

```
python2 repair_bias_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair_aug --first "bus" --second "person" --third "clock" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 3 --class_num 80
```

#### [w-bn]:  

```
python2 repair_bias_bn.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair_bn --first "bus" --second "person" --third "clock" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --replace --ratio 0.4
```

#### [w-os]:
```
python2 coco_feature_space.py --pretrained original_model/checkpoint.pth.tar --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --groupname original
python2 repair_confusion_exp_newbn_softmax.py --data_file original_test_data.npy --eta 0.8 --mode bias --first "bus" --second "person" --third "clock"
```

#### [w-loss]:
```
python2 repair_bias_exp_weighted_loss.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair_loss --first "bus" --second "person" --third "clock" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --weight 1 --class_num 80 --target_weight 0.4
```
#### [w-dbr]:  

```
python2 repair_bias_dbr.py --pretrained original_model/model_best.pth.tar --log_dir coco_bias_repair_dbr --first "bus" --second "person" --third "clock" --second "bus" --ann_dir '/local/shared/coco/annotations' --image_dir '/local/shared/coco/' --lam 0.5
```