backend_args = None
class_names = (
    'bed',
    'table',
    'sofa',
    'chair',
    'toilet',
    'desk',
    'dresser',
    'night_stand',
    'bookshelf',
    'bathtub',
)
custom_hooks = [
    dict(after_iter=True, type='EmptyCacheHook'),
]
custom_imports = dict(imports=[
    'projects.TR3D.tr3d',
])
data_root = '/home/zero/USER/HEYU/DATA/sunrgbd_matlab'
dataset_type = 'SUNRGBDDataset'
default_hooks = dict(
    checkpoint=dict(interval=-1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook'))
default_scope = 'mmdet3d'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=(
        'bed',
        'table',
        'sofa',
        'chair',
        'toilet',
        'desk',
        'dresser',
        'night_stand',
        'bookshelf',
        'bathtub',
    ))
model = dict(
    backbone=dict(
        depth=34,
        in_channels=3,
        norm='batch',
        num_planes=(
            64,
            128,
            128,
            128,
        ),
        type='TR3DMinkResNet'),
    bbox_head=dict(
        bbox_loss=dict(
            mode='diou', reduction='none', type='TR3DRotatedIoU3DLoss'),
        in_channels=128,
        label2level=[
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
        ],
        num_reg_outs=8,
        pts_center_threshold=6,
        type='TR3DHead',
        voxel_size=0.01),
    data_preprocessor=dict(type='Det3DDataPreprocessor'),
    neck=dict(
        in_channels=(
            64,
            128,
            128,
            128,
        ), out_channels=128, type='TR3DNeck'),
    test_cfg=dict(iou_thr=0.5, nms_pre=1000, score_thr=0.01),
    train_cfg=dict(),
    type='MinkSingleStage3DDetector')
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.001, type='AdamW', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = dict(
    begin=0,
    by_epoch=True,
    end=12,
    gamma=0.1,
    milestones=[
        8,
        11,
    ],
    type='MultiStepLR')
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet3d',
        ann_file='sunrgbd_infos_val.pkl',
        backend_args=None,
        box_type_3d='Depth',
        data_root='/home/zero/USER/HEYU/DATA/sunrgbd_matlab',
        metainfo=dict(
            classes=(
                'bed',
                'table',
                'sofa',
                'chair',
                'toilet',
                'desk',
                'dresser',
                'night_stand',
                'bookshelf',
                'bathtub',
            )),
        pipeline=[
            dict(
                coord_type='DEPTH',
                load_dim=6,
                shift_height=False,
                type='LoadPointsFromFile',
                use_color=True,
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                ]),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(num_points=100000, type='PointSample'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='SUNRGBDDataset'),
    num_workers=1,
    sampler=dict(_scope_='mmdet3d', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(_scope_='mmdet3d', type='IndoorMetric')
test_pipeline = [
    dict(
        coord_type='DEPTH',
        load_dim=6,
        shift_height=False,
        type='LoadPointsFromFile',
        use_color=True,
        use_dim=[
            0,
            1,
            2,
            3,
            4,
            5,
        ]),
    dict(
        flip=False,
        img_scale=(
            1333,
            800,
        ),
        pts_scale_ratio=1,
        transforms=[
            dict(num_points=100000, type='PointSample'),
        ],
        type='MultiScaleFlipAug3D'),
    dict(keys=[
        'points',
    ], type='Pack3DDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=12)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        _scope_='mmdet3d',
        dataset=dict(
            ann_file='sunrgbd_infos_train.pkl',
            backend_args=None,
            box_type_3d='Depth',
            data_root='/home/zero/USER/HEYU/DATA/sunrgbd_matlab',
            filter_empty_gt=False,
            metainfo=dict(
                classes=(
                    'bed',
                    'table',
                    'sofa',
                    'chair',
                    'toilet',
                    'desk',
                    'dresser',
                    'night_stand',
                    'bookshelf',
                    'bathtub',
                )),
            pipeline=[
                dict(
                    coord_type='DEPTH',
                    load_dim=6,
                    shift_height=False,
                    type='LoadPointsFromFile',
                    use_color=True,
                    use_dim=[
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                    ]),
                dict(type='LoadAnnotations3D'),
                dict(num_points=100000, type='PointSample'),
                dict(
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0,
                    sync_2d=False,
                    type='RandomFlip3D'),
                dict(
                    rot_range=[
                        -0.523599,
                        0.523599,
                    ],
                    scale_ratio_range=[
                        0.85,
                        1.15,
                    ],
                    shift_height=False,
                    translation_std=[
                        0.1,
                        0.1,
                        0.1,
                    ],
                    type='GlobalRotScaleTrans'),
                dict(
                    keys=[
                        'points',
                        'gt_bboxes_3d',
                        'gt_labels_3d',
                    ],
                    type='Pack3DDetInputs'),
            ],
            type='SUNRGBDDataset'),
        times=5,
        type='RepeatDataset'),
    num_workers=8,
    sampler=dict(_scope_='mmdet3d', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        coord_type='DEPTH',
        load_dim=6,
        shift_height=False,
        type='LoadPointsFromFile',
        use_color=True,
        use_dim=[
            0,
            1,
            2,
            3,
            4,
            5,
        ]),
    dict(type='LoadAnnotations3D'),
    dict(num_points=100000, type='PointSample'),
    dict(
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0,
        sync_2d=False,
        type='RandomFlip3D'),
    dict(
        rot_range=[
            -0.523599,
            0.523599,
        ],
        scale_ratio_range=[
            0.85,
            1.15,
        ],
        shift_height=False,
        translation_std=[
            0.1,
            0.1,
            0.1,
        ],
        type='GlobalRotScaleTrans'),
    dict(
        keys=[
            'points',
            'gt_bboxes_3d',
            'gt_labels_3d',
        ],
        type='Pack3DDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet3d',
        ann_file='sunrgbd_infos_val.pkl',
        backend_args=None,
        box_type_3d='Depth',
        data_root='/home/zero/USER/HEYU/DATA/sunrgbd_matlab',
        metainfo=dict(
            classes=(
                'bed',
                'table',
                'sofa',
                'chair',
                'toilet',
                'desk',
                'dresser',
                'night_stand',
                'bookshelf',
                'bathtub',
            )),
        pipeline=[
            dict(
                coord_type='DEPTH',
                load_dim=6,
                shift_height=False,
                type='LoadPointsFromFile',
                use_color=True,
                use_dim=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                ]),
            dict(
                flip=False,
                img_scale=(
                    1333,
                    800,
                ),
                pts_scale_ratio=1,
                transforms=[
                    dict(num_points=100000, type='PointSample'),
                ],
                type='MultiScaleFlipAug3D'),
            dict(keys=[
                'points',
            ], type='Pack3DDetInputs'),
        ],
        test_mode=True,
        type='SUNRGBDDataset'),
    num_workers=1,
    sampler=dict(_scope_='mmdet3d', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(_scope_='mmdet3d', type='IndoorMetric')
vis_backends = [
    dict(_scope_='mmdet3d', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet3d',
    name='visualizer',
    type='Det3DLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/zero/USER/HEYU/mmdetection3d/work_dirs/tr3d/JX_1GPU_matlab/2/'
