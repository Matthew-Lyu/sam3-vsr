#!/usr/bin/env python
"""
调试脚本 - 查看SAM3的原始输出
"""

import torch
from sam3.model_builder import build_sam3_video_predictor

# 配置
video_path = "/data2/xc/hf_cache/cambrians_vsr/10mins/00000007.mp4"
prompt_text = "teddy bear"
checkpoint_path = "~/Spatial/lwc_spatial_work/cambrains_vsr_sam3_pre_seg/sam3/checkpoints/sam3.pt"

print("=" * 80)
print("SAM3 调试脚本")
print("=" * 80)

# 初始化预测器
print(f"\n1. 初始化SAM3...")
print(f"   模型权重: {checkpoint_path}")
print(f"   GPU: 7")

predictor = build_sam3_video_predictor(
    gpus_to_use=[7],
    checkpoint_path=checkpoint_path
)

try:
    # 启动会话
    print(f"\n2. 加载视频: {video_path}")
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    print(f"   会话ID: {session_id}")

    # 添加文本提示
    print(f"\n3. 添加文本提示: '{prompt_text}'")
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=prompt_text,
        )
    )

    # 打印第0帧的响应
    print(f"\n4. 第0帧的检测结果:")
    print(f"   response keys: {response.keys()}")
    if "outputs" in response:
        outputs = response["outputs"]
        print(f"   outputs keys: {outputs.keys()}")
        if "out_probs" in outputs:
            print(f"   out_probs: {outputs['out_probs']}")
            print(f"   out_probs length: {len(outputs['out_probs'])}")
        if "out_obj_ids" in outputs:
            print(f"   out_obj_ids: {outputs['out_obj_ids']}")
        if "out_boxes_xywh" in outputs and outputs["out_boxes_xywh"] is not None:
            print(f"   out_boxes_xywh shape: {outputs['out_boxes_xywh'].shape}")
        if "out_binary_masks" in outputs and outputs["out_binary_masks"] is not None:
            print(f"   out_binary_masks shape: {outputs['out_binary_masks'].shape}")

    # 传播到几帧看看
    print(f"\n5. 传播分割到前50帧...")
    frame_count = 0
    detections_summary = []

    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        frame_idx = response["frame_index"]
        outputs = response["outputs"]

        has_detections = "out_probs" in outputs and len(outputs["out_probs"]) > 0
        if has_detections:
            max_score = max(outputs["out_probs"])
            num_detections = len(outputs["out_probs"])
            detections_summary.append((frame_idx, num_detections, max_score))
            if frame_idx < 10 or num_detections > 0:  # 显示前10帧或有检测的帧
                print(f"   Frame {frame_idx}: {num_detections} detections, max_score={max_score:.4f}")
        else:
            if frame_idx < 10:
                print(f"   Frame {frame_idx}: 0 detections")

        frame_count += 1
        if frame_count >= 50:
            break

    if detections_summary:
        print(f"\n   总结: 前50帧中有 {len(detections_summary)} 帧检测到物体")
        print(f"   最高分数: {max(d[2] for d in detections_summary):.4f}")
    else:
        print(f"\n   总结: 前50帧都没有检测到物体")

    print(f"\n6. 总结:")
    print(f"   处理了 {frame_count} 帧")

    # 关闭会话
    predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )

finally:
    predictor.shutdown()

print("\n" + "=" * 80)
print("调试完成")
print("=" * 80)
