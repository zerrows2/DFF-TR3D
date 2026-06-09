lr = 0.008
optimizer = dict(type='AdamW', lr=0.008, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[24, 32])
runner = dict(type='EpochBasedRunner', max_epochs=36)
evaluation = dict(
    interval=36,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='DEPTH',
            shift_height=False,
            load_dim=6,
            use_dim=[0, 1, 2]),
        dict(
            type='DefaultFormatBundle3D',
            class_names=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                         'dresser', 'night_stand', 'bookshelf', 'bathtub'),
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
model = dict(
    type='VoteNetFF',
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    backbone=dict(
        type='RgbdPointNet2SASSG',
        in_channels=4,
        num_points=(2048, 1024, 512, 256),
        radius=(0.2, 0.4, 0.8, 1.2),
        num_samples=(64, 32, 16, 16),
        sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),
                     (128, 128, 256)),
        fp_channels=((256, 256), (256, 256)),
        norm_cfg=dict(type='BN2d'),
        sa_cfg=dict(
            type='PointSAModule',
            pool_mod='max',
            use_xyz=True,
            normalize_xyz=True)),
    bbox_head=dict(
        type='VoteHead',
        num_classes=10,
        bbox_coder=dict(
            type='PartialBinBasedBBoxCoder',
            num_sizes=10,
            num_dir_bins=12,
            with_rot=True,
            mean_sizes=[[2.114256, 1.6203, 0.927272],
                        [0.791118, 1.279516, 0.718182],
                        [0.923508, 1.867419, 0.845495],
                        [0.591958, 0.552978, 0.827272],
                        [0.699104, 0.454178, 0.75625],
                        [0.69519, 1.346299, 0.736364],
                        [0.528526, 1.002642, 1.172878],
                        [0.500618, 0.632163, 0.683424],
                        [0.404671, 1.071108, 1.688889],
                        [0.76584, 1.398258, 0.472728]]),
        vote_module_cfg=dict(
            in_channels=256,
            vote_per_seed=1,
            gt_per_seed=3,
            conv_channels=(256, 256),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d'),
            norm_feats=True,
            vote_loss=dict(
                type='ChamferDistance',
                mode='l1',
                reduction='none',
                loss_dst_weight=10.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModule',
            num_point=256,
            radius=0.3,
            num_sample=16,
            mlp_channels=[256, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True),
        pred_layer_cfg=dict(
            in_channels=128, shared_conv_channels=(128, 128), bias=True),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            class_weight=[0.2, 0.8],
            reduction='sum',
            loss_weight=5.0),
        center_loss=dict(
            type='ChamferDistance',
            mode='l2',
            reduction='sum',
            loss_src_weight=10.0,
            loss_dst_weight=10.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=10.0),
        size_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss',
            reduction='sum',
            loss_weight=3.3333333333333335),
        semantic_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(
        pos_distance_thr=0.3, neg_distance_thr=0.6, sample_mod='vote'),
    test_cfg=dict(
        sample_mod='seed',
        nms_thr=0.25,
        score_thr=0.05,
        per_class_proposal=True))
dataset_type = 'SUNRGBDDataset'
data_root = '/home/zero/USER/HEYU/DATA/sunrgbd_matlab/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations3D'),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 504), (1333, 528), (1333, 552),
                   (1333, 576), (1333, 600)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='PointSample', num_points=20000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                     'dresser', 'night_stand', 'bookshelf', 'bathtub')),
    dict(
        type='Collect3D',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 600),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='Resize', multiscale_mode='value', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.5),
            dict(type='PointSample', num_points=20000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                             'dresser', 'night_stand', 'bookshelf', 'bathtub'),
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=6,
        use_dim=[0, 1, 2]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                     'dresser', 'night_stand', 'bookshelf', 'bathtub'),
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='SUNRGBDDataset',
            modality=dict(use_camera=True, use_lidar=True),
            data_root='/home/zero/USER/HEYU/DATA/sunrgbd_matlab/',
            ann_file=
            '/home/zero/USER/HEYU/DATA/sunrgbd_matlab/sunrgbd_infos_train.pkl',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='DEPTH',
                    shift_height=True,
                    load_dim=6,
                    use_dim=[0, 1, 2]),
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations3D'),
                dict(
                    type='Resize',
                    img_scale=[(1333, 480), (1333, 504), (1333, 528),
                               (1333, 552), (1333, 576), (1333, 600)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.523599, 0.523599],
                    scale_ratio_range=[0.85, 1.15],
                    shift_height=True),
                dict(type='PointSample', num_points=20000),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.0),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=('bed', 'table', 'sofa', 'chair', 'toilet',
                                 'desk', 'dresser', 'night_stand', 'bookshelf',
                                 'bathtub')),
                dict(
                    type='Collect3D',
                    keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            classes=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                     'dresser', 'night_stand', 'bookshelf', 'bathtub'),
            filter_empty_gt=False,
            box_type_3d='Depth')),
    val=dict(
        type='SUNRGBDDataset',
        modality=dict(use_camera=True, use_lidar=True),
        data_root='/home/zero/USER/HEYU/DATA/sunrgbd_matlab/',
        ann_file=
        '/home/zero/USER/HEYU/DATA/sunrgbd_matlab/sunrgbd_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=True,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 600),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(
                        type='RandomFlip3D',
                        sync_2d=False,
                        flip_ratio_bev_horizontal=0.5),
                    dict(type='PointSample', num_points=20000),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=('bed', 'table', 'sofa', 'chair', 'toilet',
                                     'desk', 'dresser', 'night_stand',
                                     'bookshelf', 'bathtub'),
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'img'])
                ])
        ],
        classes=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                 'night_stand', 'bookshelf', 'bathtub'),
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type='SUNRGBDDataset',
        modality=dict(use_camera=True, use_lidar=True),
        data_root='/home/zero/USER/HEYU/DATA/sunrgbd_matlab/',
        ann_file=
        '/home/zero/USER/HEYU/DATA/sunrgbd_matlab/sunrgbd_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=True,
                load_dim=6,
                use_dim=[0, 1, 2]),
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 600),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        multiscale_mode='value',
                        keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(
                        type='RandomFlip3D',
                        sync_2d=False,
                        flip_ratio_bev_horizontal=0.5),
                    dict(type='PointSample', num_points=20000),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=('bed', 'table', 'sofa', 'chair', 'toilet',
                                     'desk', 'dresser', 'night_stand',
                                     'bookshelf', 'bathtub'),
                        with_label=False),
                    dict(type='Collect3D', keys=['points', 'img'])
                ])
        ],
        classes=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
                 'night_stand', 'bookshelf', 'bathtub'),
        test_mode=True,
        box_type_3d='Depth'))
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/zero/USER/HEYU/tr3d/work_dirs/DFF_RFDETR_VoteNet_FF/4'
load_from = '/home/zero/USER/HEYU/MODEL/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210819_225618-62eba6ce.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
gpu_ids = [0]
