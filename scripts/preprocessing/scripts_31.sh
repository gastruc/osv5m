python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=4096 model.name=classif_hier_nll class_name=city model/loss=cls_hier mode=traineval max_epochs=30

python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=4096 model.name=classif_hier_nll4e-4 class_name=city model/loss=cls_hier mode=traineval max_epochs=30 model.optimizer.optim.lr=0.0004

python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=4096 model.name=classif_hier_nll1e-4 class_name=city model/loss=cls_hier mode=traineval max_epochs=30 model.optimizer.optim.lr=0.0001

python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=4096 model.name=classif_hier_nll1e-3 class_name=city model/loss=cls_hier mode=traineval max_epochs=30 model.optimizer.optim.lr=0.001

python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=4096 model.name=classif_hier_nll4e-3 class_name=city model/loss=cls_hier mode=traineval max_epochs=30 model.optimizer.optim.lr=0.004

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_all_aux_fin_lr5e-6' aux_data=['land_cover','drive_side','climate','soil','dist_sea'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=5e-6

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_all_aux_fin_lr5e-6' aux_data=['land_cover','drive_side','climate','soil','dist_sea'] model.optimizer.unfreeze_lr=True mode='eval'

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_all_aux_fin_lr2e-5' aux_data=['land_cover','drive_side','climate','soil','dist_sea'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_all_aux_fin_lr2e-5' aux_data=['land_cover','drive_side','climate','soil','dist_sea'] model.optimizer.unfreeze_lr=True mode='eval'

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_land_fin' aux_data=['land_cover'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_drive_fin' aux_data=['drive_side'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_climate_fin' aux_data=['climate'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_soil_fin' aux_data=['soil'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_sea_fin' aux_data=['dist_sea'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_land_fin' aux_data=['land_cover'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5 mode='eval'

python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_drive_fin' aux_data=['drive_side'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5 mode='eval'

python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_climate_fin' aux_data=['climate'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5 mode='eval'

python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_soil_fin' aux_data=['soil'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5 mode='eval'

python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_sea_fin' aux_data=['dist_sea'] model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5 mode='eval'


#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=1024 model.name=Regression_contrastive_region_corr_fine class_name=region model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=4e-5

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=1024 model.name=Regression_contrastive_region_corr_fine class_name=region model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=4e-5 mode='eval'

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=1024 model.name=Regression_contrastive_sub-region_corr_fine class_name=sub-region model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=4e-5

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=1024 model.name=Regression_contrastive_sub-region_corr_fine class_name=sub-region model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=4e-5 mode='eval'

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=1024 model.name=Regression_contrastive_city_corr_fine class_name=city model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=4e-5

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=1024 model.name=Regression_contrastive_city_corr_fine class_name=city model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=4e-5 mode='eval'

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=1024 model.name=Regression_contrastive_country_corr_fine class_name=country model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=4e-5

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=1024 model.name=Regression_contrastive_country_corr_fine class_name=country model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=4e-5 mode='eval'

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_unfrozen_diff_lr_1e-5' model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=1e-5

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_unfrozen_diff_lr_1e-5' model.optimizer.unfreeze_lr=True mode='eval'

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_unfrozen_diff_lr_2e-5' model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_unfrozen_diff_lr_2e-5' model.optimizer.unfreeze_lr=True mode='eval'

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_unfrozen_diff_lr_5e-6' model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=5e-6

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_unfrozen_diff_lr_5e-6' model.optimizer.unfreeze_lr=True mode='eval'

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_unfrozen_diff_last_lr_5e-6' model.optimizer.unfreeze_lr=True model.optimizer.diff_backbone_last=True

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_unfrozen_diff_last_lr_5e-6' model.optimizer.unfreeze_lr=True mode='eval'

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_unfrozen_diff_last_lr_1e-5' model.optimizer.unfreeze_lr=True model.optimizer.diff_backbone_last=True model.optimizer.backbone_lr=1e-5

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_unfrozen_diff_last_lr_1e-5' model.optimizer.unfreeze_lr=True mode='eval'

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=256 model.name='Regression_contrastive_country' class_name='country' max_epochs=8

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=256 model.name='Regression_contrastive_region' class_name='region' max_epochs=8

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=256 model.name='Regression_contrastive_sub-region' class_name='sub-region' max_epochs=8

#python train.py exp=contrastive_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=256 model.name='Regression_contrastive_city' class_name='city' max_epochs=8

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_last_block_diff_lr' model/network=last_block_backbone model.optimizer.unfreeze_lr=True

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_last_block_diff_lr' model/network=last_block_backbone model.optimizer.unfreeze_lr=True mode='eval'

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_land_l' aux_data=['land_cover'] model/network=last_block_backbone model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_drive_l' aux_data=['drive_side'] model/network=last_block_backbone model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_climate_l' aux_data=['climate'] model/network=last_block_backbone model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_soil_l' aux_data=['soil'] model/network=last_block_backbone model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_sea_l' aux_data=['dist_sea'] model/network=last_block_backbone model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_land_l' aux_data=['land_cover'] model/network=last_block_backbone model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5 mode='eval'

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_drive_l' aux_data=['drive_side'] model/network=last_block_backbone model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5 mode='eval'

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_climate_l' aux_data=['climate'] model/network=last_block_backbone model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5 mode='eval'

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_soil_l' aux_data=['soil'] model.optimizer.unfreeze_lr=True model/network=last_block_backbone model.optimizer.backbone_lr=2e-5 mode='eval'

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_sea_l' aux_data=['dist_sea'] model.optimizer.unfreeze_lr=True model/network=last_block_backbone model.optimizer.backbone_lr=2e-5 mode='eval'

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_all_aux_l' aux_data=['land_cover', 'drive_side', 'climate', 'soil', 'dist_sea'] model/network=last_block_backbone model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr2e-5

#python train.py exp=multi computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='Regression_all_aux_l' aux_data=['land_cover', 'drive_side', 'climate', 'soil', 'dist_sea'] model/network=last_block_backbone model.optimizer.unfreeze_lr=True model.optimizer.backbone_lr=2e-5 mode='eval'

#python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model/network=last_block_backbone model.name='classif_country' class_name='country'

#python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model/network=last_block_backbone model.name='classif_city' class_name='city'

#python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model/network=last_block_backbone model.name='classif_region' class_name='region'

#python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model/network=last_block_backbone model.name='classif_sub-region' class_name='sub-region'

#python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model/network=last_block_backbone model.name='classif_country' class_name='country' mode='eval'

#python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model/network=last_block_backbone model.name='classif_city' class_name='city' mode='eval'

#python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model/network=last_block_backbone model.name='classif_region' class_name='region' mode='eval'

#python train.py exp=clip_cls computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model/network=last_block_backbone model.name='classif_sub-region' class_name='sub-region' mode='eval'

#python train.py exp=clip_reg_lora computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_test_lora_lr_210-6' model.optimizer.lora_lr=0.000002 model.optimizer.optim.lr=0.0002

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_unfrozen_lr_10-5' mode='eval'

#python train.py exp=clip_reg_lora computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_test_lora_lr_210-5' mode='eval'

#python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_base2' mode='eval'

#python train.py exp=clip_reg_unfrozen computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_last_block' model/network=last_block_backbone mode='eval'
