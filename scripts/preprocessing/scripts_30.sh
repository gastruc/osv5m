python train.py exp=clip_hybrid computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='hybrid_l2_quadtree_50000' class_name=quadtree_10_50000

python train.py exp=clip_hybrid computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='hybrid_l2_quadtree_50000' class_name=quadtree_10_50000 mode='eval'

python train.py exp=clip_hybrid computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='hybrid_l23_quadtree_25000' class_name=quadtree_10_25000

python train.py exp=clip_hybrid computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='hybrid_l23_quadtree_25000' class_name=quadtree_10_25000 mode='eval'

python train.py exp=clip_hybrid computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='hybrid_l2_quadtree_5000' class_name=quadtree_10_5000

python train.py exp=clip_hybrid computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='hybrid_l2_quadtree_5000' class_name=quadtree_10_5000 mode='eval'

python train.py exp=clip_hybrid computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='hybrid_l2_quadtree_12500' class_name=quadtree_10_12500

python train.py exp=clip_hybrid computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='hybrid_l2_quadtree_12500' class_name=quadtree_10_12500 mode='eval'

python train.py exp=clip_hybrid computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='hybrid_l2_quadtree_1000' class_name=quadtree_10_1000

python train.py exp=clip_hybrid computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=2048 model.name='hybrid_l2_quadtree_1000' class_name=quadtree_10_1000 mode='eval'

#python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_clip_L' model/network/backbone=clip_L_14 mode='eval'

#python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_dinov2_vitl14' model/network/backbone=dinov2_vitl14 mode='eval'

#python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_clip_L_DataComp' model/network/backbone=clip_L_14_DataComp mode='eval'

#python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_openclip_L14' model/network/backbone=openclip_L_14

#python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_openclip_L14' model/network/backbone=openclip_L_14 mode='eval'

#python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_streetclip_L14' model/network/backbone=streetclip

#python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_streetclip_L14' model/network/backbone=streetclip mode='eval'

#python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_openclip_bigG14' model/network/backbone=openclip_bigG_14

#python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_openclip_bigG14' model/network/backbone=openclip_bigG_14 mode='eval'

python train.py exp=clip_reg computer=a100 computer.devices=3 computer.num_workers=16 dataset.global_batch_size=512 model.name='Regression_scratch' model/network/backbone=scratch_B_32 max_epochs=100
