voxel_size = 0.01
n_points = 100000
model = dict(
    type='TR3DFF3DDetector',
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
        type='MinkFFResNet',
        in_channels=3,
        max_channels=128,
        depth=34,
        norm='batch'),
    neck=dict(
        type='TR3DNeck', in_channels=(64, 128, 128, 128), out_channels=128),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=8,
        n_classes=10,
        voxel_size=0.01,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=[1, 1, 1, 0, 0, 1, 0, 0, 1, 0]),
        bbox_loss=dict(type='RotatedIoU3DLoss', mode='diou',
                       reduction='none')),
    voxel_size=0.01,
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 12, 16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/star/user/HEYU/tr3d/work_dirs/epoch/Dynamic/epoch119/4/'
load_from = '/home/star/user/HEYU/MODEL/imvotenet_faster_rcnn_r50_fpn_2x4_sunrgbd-3d-10class_20210323_173222-cad62aeb.pth'
resume_from = None
workflow = [('train', 1)]
dataset_type = 'SUNRGBDDataset'
data_root = '/home/star/user/HEYU/DATA/sunrgbd/'
class_names = ('bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser',
               'night_stand', 'bookshelf', 'bathtub')
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
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
    dict(type='PointSample', num_points=100000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
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
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
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
            dict(type='PointSample', num_points=100000),
            dict(
                type='DefaultFormatBundle3D',
                class_names=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                             'dresser', 'night_stand', 'bookshelf', 'bathtub'),
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
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
            data_root='/home/star/user/HEYU/DATA/sunrgbd/',
            ann_file=
            '/home/star/user/HEYU/DATA/sunrgbd/sunrgbd_infos_train.pkl',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='DEPTH',
                    shift_height=False,
                    use_color=True,
                    load_dim=6,
                    use_dim=[0, 1, 2, 3, 4, 5]),
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
                dict(type='PointSample', num_points=100000),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.0),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.523599, 0.523599],
                    scale_ratio_range=[0.85, 1.15],
                    translation_std=[0.1, 0.1, 0.1],
                    shift_height=False),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=('bed', 'table', 'sofa', 'chair', 'toilet',
                                 'desk', 'dresser', 'night_stand', 'bookshelf',
                                 'bathtub')),
                dict(
                    type='Collect3D',
                    keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            filter_empty_gt=False,
            classes=('bed', 'table', 'sofa', 'chair', 'toilet', 'desk',
                     'dresser', 'night_stand', 'bookshelf', 'bathtub'),
            box_type_3d='Depth')),
    val=dict(
        type='SUNRGBDDataset',
        modality=dict(use_camera=True, use_lidar=True),
        data_root='/home/star/user/HEYU/DATA/sunrgbd/',
        ann_file='/home/star/user/HEYU/DATA/sunrgbd/sunrgbd_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
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
                    dict(type='PointSample', num_points=100000),
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
        data_root='/home/star/user/HEYU/DATA/sunrgbd/',
        ann_file='/home/star/user/HEYU/DATA/sunrgbd/sunrgbd_infos_val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
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
                    dict(type='PointSample', num_points=100000),
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
gpu_ids = [0]
