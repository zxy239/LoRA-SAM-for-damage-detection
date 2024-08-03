auto_scale_lr = dict(base_batch_size=4, enable=False)
backend_args = None
base_lr = 0.0001
batch_augments = [
    dict(
        img_pad_value=0,
        mask_pad_value=0,
        pad_mask=True,
        pad_seg=False,
        size=(
            1024,
            1024,
        ),
        type='BatchFixedSizePad'),
]
batch_size = 4
batch_size_per_gpu = 4
code_root = ''
crop_size = (
    1024,
    1024,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'mmdet.rsprompter',
    ])
data_preprocessor = dict(
    batch_augments=[
        dict(
            img_pad_value=0,
            mask_pad_value=0,
            pad_mask=True,
            pad_seg=False,
            size=(
                1024,
                1024,
            ),
            type='BatchFixedSizePad'),
    ],
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_mask=True,
    pad_size_divisor=32,
    std=[
        58.395,
        57.120000000000005,
        57.375,
    ],
    type='DetDataPreprocessor')
data_root = ''
dataset_type = 'MasonryDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,
        rule='greater',
        save_best='coco/segm_mAP',
        save_last=True,
        type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
hf_sam_pretrain_ckpt_path = 'tools/rsprompter/pretrain_models/pytorch_model.bin'
hf_sam_pretrain_name = 'facebook/sam-vit-huge'
indices = None
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 400
model = dict(
    backbone=dict(
        extra_config=dict(output_hidden_states=True),
        hf_pretrain_name='facebook/sam-vit-huge',
        init_cfg=dict(
            checkpoint='tools/rsprompter/pretrain_models/pytorch_model.bin',
            type='Pretrained'),
        peft_config=dict(
            bias='none',
            lora_alpha=32,
            lora_dropout=0.05,
            peft_type='LORA',
            r=16,
            target_modules=[
                'qkv',
            ]),
        type='RSSamVisionEncoder'),
    data_preprocessor=dict(
        batch_augments=[
            dict(
                img_pad_value=0,
                mask_pad_value=0,
                pad_mask=True,
                pad_seg=False,
                size=(
                    1024,
                    1024,
                ),
                type='BatchFixedSizePad'),
        ],
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.120000000000005,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        feature_aggregator=dict(
            hidden_channels=32,
            in_channels='facebook/sam-vit-huge',
            out_channels=256,
            select_layers=range(32, 33),
            type='RSFeatureAggregator'),
        feature_spliter=dict(
            backbone_channel=1280,
            in_channels=[
                256,
                256,
                256,
                256,
            ],
            norm_cfg=dict(requires_grad=True, type='LN2d'),
            num_outs=5,
            out_channels=256,
            type='RSSimpleFPN'),
        type='RSFPN'),
    panoptic_fusion_head=dict(
        init_cfg=None,
        loss_panoptic=None,
        num_stuff_classes=0,
        num_things_classes=3,
        type='MaskFormerFusionHead'),
    panoptic_head=dict(
        enforce_decoder_input_project=False,
        feat_channels=256,
        in_channels=[
            256,
            256,
            256,
            256,
            256,
        ],
        loss_cls=dict(
            class_weight=[
                1.0,
                1.0,
                1.0,
                0.1,
            ],
            loss_weight=2.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=False),
        loss_dice=dict(
            activate=True,
            eps=1.0,
            loss_weight=5.0,
            naive_dice=True,
            reduction='mean',
            type='DiceLoss',
            use_sigmoid=True),
        loss_mask=dict(
            loss_weight=5.0,
            reduction='mean',
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_queries=80,
        num_stuff_classes=0,
        num_things_classes=3,
        num_transformer_feat_level=3,
        out_channels=256,
        pixel_decoder=dict(
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                layer_cfg=dict(
                    ffn_cfg=dict(
                        act_cfg=dict(inplace=True, type='ReLU'),
                        embed_dims=256,
                        feedforward_channels=1024,
                        ffn_drop=0.0,
                        num_fcs=2),
                    self_attn_cfg=dict(
                        batch_first=True,
                        dropout=0.0,
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4)),
                num_layers=3),
            norm_cfg=dict(num_groups=32, type='GN'),
            num_outs=3,
            positional_encoding=dict(normalize=True, num_feats=128),
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='MSDeformAttnPixelDecoder'),
        positional_encoding=dict(normalize=True, num_feats=128),
        transformer_decoder=dict(
            init_cfg=None,
            layer_cfg=dict(
                cross_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8),
                ffn_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    embed_dims=256,
                    feedforward_channels=2048,
                    ffn_drop=0.0,
                    num_fcs=2),
                self_attn_cfg=dict(
                    batch_first=True, dropout=0.0, embed_dims=256,
                    num_heads=8)),
            num_layers=9,
            return_intermediate=True),
        type='Mask2FormerHead'),
    test_cfg=dict(
        filter_low_score=True,
        instance_on=True,
        iou_thr=0.8,
        max_per_image=80,
        panoptic_on=False,
        semantic_on=False),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='ClassificationCost', weight=2.0),
                dict(
                    type='CrossEntropyLossCost', use_sigmoid=True, weight=5.0),
                dict(eps=1.0, pred_act=True, type='DiceCost', weight=5.0),
            ],
            type='HungarianAssigner'),
        importance_sample_ratio=0.75,
        num_points=12544,
        oversample_ratio=3.0,
        sampler=dict(type='MaskPseudoSampler')),
    type='SAMSegMask2Former')
num_classes = 1
num_queries = 80
num_stuff_classes = 0
num_things_classes = 3
num_workers = 36
optim_wrapper = dict(
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.05),
    type='DeepSpeedOptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=50, start_factor=0.001, type='LinearLR'),
    dict(
        T_max=400,
        begin=1,
        by_epoch=True,
        end=400,
        eta_min=1.0000000000000001e-07,
        type='CosineAnnealingLR'),
]
persistent_workers = True
resume = False
runner_type = 'FlexibleRunner'
strategy = dict(
    fp16=dict(
        auto_cast=False,
        enabled=True,
        fp16_master_weights_and_grads=False,
        hysteresis=2,
        initial_scale_power=15,
        loss_scale=0,
        loss_scale_window=500,
        min_loss_scale=1),
    gradient_clipping=0.1,
    inputs_to_half=[
        'inputs',
    ],
    type='DeepSpeedStrategy',
    zero_optimization=dict(
        allgather_bucket_size=200000000.0,
        allgather_partitions=True,
        contiguous_gradients=True,
        overlap_comm=True,
        reduce_bucket_size='auto',
        reduce_scatter=True,
        stage=2))
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='data/Masonry/valid/_annotations.coco.json',
        backend_args=None,
        data_prefix=dict(img='data/Masonry/valid'),
        data_root='',
        indices=None,
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    103.53,
                    116.28,
                    123.675,
                ), masks=0),
                size=(
                    1024,
                    1024,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='MasonryDataset'),
    drop_last=False,
    num_workers=36,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1024,
        1024,
    ), type='Resize'),
    dict(
        pad_val=dict(img=(
            103.53,
            116.28,
            123.675,
        ), masks=0),
        size=(
            1024,
            1024,
        ),
        type='Pad'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=400, type='EpochBasedTrainLoop', val_interval=2)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='data/Masonry/train/_annotations.coco.json',
        backend_args=None,
        data_prefix=dict(img='data/Masonry/train'),
        data_root='',
        indices=None,
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                resize_type='Resize',
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    1024,
                    1024,
                ),
                crop_type='absolute',
                recompute_bbox=True,
                type='RandomCrop'),
            dict(
                by_mask=True,
                min_gt_bbox_wh=(
                    1e-05,
                    1e-05,
                ),
                type='FilterAnnotations'),
            dict(type='PackDetInputs'),
        ],
        type='MasonryDataset'),
    num_workers=36,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        resize_type='Resize',
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        crop_size=(
            1024,
            1024,
        ),
        crop_type='absolute',
        recompute_bbox=True,
        type='RandomCrop'),
    dict(
        by_mask=True,
        min_gt_bbox_wh=(
            1e-05,
            1e-05,
        ),
        type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='data/Masonry/valid/_annotations.coco.json',
        backend_args=None,
        data_prefix=dict(img='data/Masonry/valid'),
        data_root='',
        indices=None,
        pipeline=[
            dict(backend_args=None, to_float32=True, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1024,
                1024,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    103.53,
                    116.28,
                    123.675,
                ), masks=0),
                size=(
                    1024,
                    1024,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='MasonryDataset'),
    drop_last=False,
    num_workers=36,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        init_kwargs=dict(
            name='LoRASAM-mask2former-masonry0107(80)_e400',
            project='rsprompter-new'),
        type='WandbVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            init_kwargs=dict(
                name='LoRASAM-mask2former-masonry0107(80)_e400',
                project='rsprompter-new'),
            type='WandbVisBackend'),
    ])
work_dir = './work_dirs_new/samseg-mask2former-masonry0107(80)'
