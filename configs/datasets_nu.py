albu_train_transforms = [
    dict(type='Flip',p=0.5),
    dict(type='RandomRotate90',p=1.0),
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(type='Resize',
         img_scale=[(1600, 400), (1600, 1400)], 
         multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(type='BboxParams',format='pascal_voc',label_fields=['gt_labels'], min_visibility=0.0,filter_lost_elements=True),
        keymap={'img': 'image','gt_masks': 'masks','gt_bboxes': 'bboxes','gt_semantic_seg':'mask'},
        update_pad_shape=False,
        skip_img_without_anno=True),    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='SegRescale', scale_factor=1 / 8),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_semantic_seg']),
]


test_pipeline = [
     dict(type='LoadImageFromFile'),
     dict(
         type='MultiScaleFlipAug',
         img_scale=[(600,400), (1440, 1080),(1600,1200),],
         flip=True,
       #  flip_direction=["horizontal", "vertical"],
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='RandomFlip', flip_ratio=0.5),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect', keys=['img']),
         ])
 ]

dataset_type = 'CocoDataset'
classes = ('cell',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='./datasets/trainval/train_nu_t298.json',
        img_prefix='./datasets/trainval/x',
        seg_prefix = './datasets/trainval/nu_seg',,        
        pipeline = train_pipeline,
        ),
    val=dict(
        type=dataset_type,
        classes=classes,
        pipeline = test_pipeline,
        ann_file='./datasets/trainval/val_nu_t200.json',
        img_prefix='./datasets/trainval/x',
    )
    test=dict(
        type=dataset_type,
        classes=classes,
        pipeline = test_pipeline,
        ann_file='./datasets/trainval/val_nu_t200.json',
        img_prefix='./datasets/trainval/x',))
evaluation = dict(metric=['bbox', 'segm'])