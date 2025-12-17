#!/usr/bin/env python
"""
视频预分割脚本 - 使用SAM3对视频中的特定目标进行分割和追踪

功能：
- 使用文本提示（如"teddy bear"）在视频中检测和追踪目标物体
- 输出JSON文件记录目标物体出现的所有帧及其位置信息
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_predictor


def segment_video_with_text_prompt(
    video_path: str,
    prompt_text: str,
    output_json_path: str,
    gpus_to_use: list = None,
    score_threshold: float = 0.5,
    checkpoint_path: str = None,
):
    """
    使用SAM3对视频进行文本提示的分割

    Args:
        video_path: 视频文件路径（mp4）或JPEG帧文件夹路径
        prompt_text: 文本提示，如"teddy bear", "person", "car"等
        output_json_path: 输出JSON文件路径
        gpus_to_use: 使用的GPU列表，如[0, 1]，None表示使用所有可用GPU
        score_threshold: 分数阈值，低于此分数的检测结果将被过滤
        checkpoint_path: 本地模型权重路径，如果不提供则从HuggingFace下载
    """

    # 初始化预测器
    if gpus_to_use is None:
        gpus_to_use = range(torch.cuda.device_count())

    print(f"正在初始化SAM3视频预测器，使用GPU: {list(gpus_to_use)}")
    if checkpoint_path:
        print(f"使用本地模型权重: {checkpoint_path}")
        predictor = build_sam3_video_predictor(
            gpus_to_use=gpus_to_use,
            checkpoint_path=checkpoint_path
        )
    else:
        print("从HuggingFace下载模型权重...")
        predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    try:
        # 启动会话
        print(f"正在加载视频: {video_path}")
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]

        # 先在frame 0添加prompt获取总帧数
        print("正在获取视频信息...")
        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=0,
                text=prompt_text,
            )
        )

        frame_count = 0
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            frame_count = response["frame_index"] + 1

        print(f"视频总帧数: {frame_count}")
        print(f"正在每帧独立检测目标: '{prompt_text}'...")

        # 在每一帧上独立检测
        outputs_per_frame = {}
        for frame_idx in tqdm(range(frame_count), desc="检测中"):
            # 重置会话
            predictor.handle_request(
                request=dict(
                    type="reset_session",
                    session_id=session_id,
                )
            )

            # 在当前帧添加prompt
            response = predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=frame_idx,
                    text=prompt_text,
                )
            )

            outputs_per_frame[frame_idx] = response["outputs"]

        # 处理输出结果
        print("正在处理输出结果...")
        video_name = Path(video_path).stem
        frames_with_object = []
        frame_annotations = {}

        for frame_idx, outputs in outputs_per_frame.items():
            # 提取分数、边界框、masks等信息
            # SAM3的输出键名是 out_probs, out_obj_ids, out_boxes_xywh, out_binary_masks
            if "out_probs" not in outputs or len(outputs["out_probs"]) == 0:
                continue

            # 过滤低分数的检测结果
            valid_indices = [
                i for i, score in enumerate(outputs["out_probs"])
                if score >= score_threshold
            ]

            if not valid_indices:
                continue

            frames_with_object.append(frame_idx)

            objects = []
            for idx in valid_indices:
                obj_info = {
                    "obj_id": int(outputs["out_obj_ids"][idx]) if "out_obj_ids" in outputs else idx,
                    "score": float(outputs["out_probs"][idx]),
                }

                # 添加边界框信息（如果存在）
                # out_boxes_xywh 已经是 [x, y, w, h] 格式
                if "out_boxes_xywh" in outputs and outputs["out_boxes_xywh"] is not None:
                    box = outputs["out_boxes_xywh"][idx]
                    obj_info["bbox"] = [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ]

                # 添加mask面积信息（如果存在）
                if "out_binary_masks" in outputs and outputs["out_binary_masks"] is not None:
                    mask = outputs["out_binary_masks"][idx]
                    obj_info["mask_area"] = int(mask.sum())

                objects.append(obj_info)

            frame_annotations[str(frame_idx)] = {
                "frame_idx": frame_idx,
                "objects": objects,
            }

        # 构建最终的JSON输出
        result = {
            "video_path": video_path,
            "video_name": video_name,
            "prompt_text": prompt_text,
            "score_threshold": score_threshold,
            "total_frames": frame_count,
            "num_frames_with_object": len(frames_with_object),
            "frames_with_object": sorted(frames_with_object),
            "frame_annotations": frame_annotations,
        }

        # 保存JSON文件
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {output_json_path}")
        print(f"总帧数: {frame_count}")
        print(f"检测到目标的帧数: {len(frames_with_object)}")
        print(f"检测率: {len(frames_with_object) / frame_count * 100:.2f}%")

        # 关闭会话
        predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

    finally:
        # 清理资源
        predictor.shutdown()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="使用SAM3对视频进行目标物体预分割"
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="视频文件路径（.mp4）或JPEG帧文件夹路径",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        required=True,
        help="文本提示，描述要分割的目标物体，如'teddy bear', 'person', 'car'等",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出JSON文件路径",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="使用的GPU ID，用逗号分隔，如'0,1'。默认使用所有可用GPU",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="分数阈值，低于此分数的检测结果将被过滤（默认: 0.5）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="本地模型权重路径（如 checkpoints/sam3.pt），如果不提供则从HuggingFace下载",
    )

    args = parser.parse_args()

    # 解析GPU列表
    gpus_to_use = None
    if args.gpus is not None:
        gpus_to_use = [int(gpu_id) for gpu_id in args.gpus.split(",")]

    # 执行分割
    segment_video_with_text_prompt(
        video_path=args.video_path,
        prompt_text=args.prompt_text,
        output_json_path=args.output,
        gpus_to_use=gpus_to_use,
        score_threshold=args.score_threshold,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
