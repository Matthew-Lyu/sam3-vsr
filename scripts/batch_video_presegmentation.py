#!/usr/bin/env python
"""
批量视频预分割脚本 - 处理整个数据集中的多个视频

功能：
- 批量处理文件夹中的所有视频
- 支持断点续传（跳过已处理的视频）
- 自动保存每个视频的分割结果为JSON文件
"""

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_predictor


def process_single_video(
    predictor,
    video_path: str,
    prompt_text: str,
    output_json_path: str,
    score_threshold: float = 0.5,
):
    """
    处理单个视频

    Args:
        predictor: SAM3视频预测器
        video_path: 视频文件路径
        prompt_text: 文本提示
        output_json_path: 输出JSON文件路径
        score_threshold: 分数阈值
    """
    try:
        # 启动会话
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]

        # 先在frame 0添加prompt获取总帧数
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

        # 在每一帧上独立检测
        outputs_per_frame = {}
        for frame_idx in tqdm(range(frame_count), desc="检测中", leave=False):
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
        video_name = Path(video_path).stem
        frames_with_object = []
        frame_annotations = {}

        for frame_idx, outputs in outputs_per_frame.items():
            # SAM3的输出键名是 out_probs, out_obj_ids, out_boxes_xywh, out_binary_masks
            if "out_probs" not in outputs or len(outputs["out_probs"]) == 0:
                continue

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

                # out_boxes_xywh 已经是 [x, y, w, h] 格式
                if "out_boxes_xywh" in outputs and outputs["out_boxes_xywh"] is not None:
                    box = outputs["out_boxes_xywh"][idx]
                    obj_info["bbox"] = [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ]

                if "out_binary_masks" in outputs and outputs["out_binary_masks"] is not None:
                    mask = outputs["out_binary_masks"][idx]
                    obj_info["mask_area"] = int(mask.sum())

                objects.append(obj_info)

            frame_annotations[str(frame_idx)] = {
                "frame_idx": frame_idx,
                "objects": objects,
            }

        # 构建JSON输出
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

        # 关闭会话
        predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

        return {
            "success": True,
            "video_name": video_name,
            "total_frames": frame_count,
            "frames_with_object": len(frames_with_object),
        }

    except Exception as e:
        return {
            "success": False,
            "video_name": Path(video_path).stem,
            "error": str(e),
        }


def batch_process_videos(
    video_dir: str,
    prompt_text: str,
    output_dir: str,
    gpus_to_use: list = None,
    score_threshold: float = 0.5,
    resume: bool = True,
    video_pattern: str = "*.mp4",
    config_json: str = None,
    checkpoint_path: str = None,
):
    """
    批量处理视频文件夹

    Args:
        video_dir: 视频文件夹路径
        prompt_text: 文本提示（如果config_json为None时使用）
        output_dir: 输出文件夹路径
        gpus_to_use: 使用的GPU列表
        score_threshold: 分数阈值
        resume: 是否跳过已处理的视频
        video_pattern: 视频文件匹配模式，如"*.mp4"
        config_json: JSON配置文件路径，包含视频名称到目标物体的映射
        checkpoint_path: 本地模型权重路径
    """
    # 如果提供了配置文件，读取视频-目标物体映射
    video_to_prompt = {}
    if config_json:
        print(f"正在从配置文件读取视频-目标物体映射: {config_json}")
        with open(config_json, 'r', encoding='utf-8') as f:
            video_to_prompt = json.load(f)
        print(f"配置文件包含 {len(video_to_prompt)} 个视频的目标物体映射")

    # 获取所有视频文件
    video_dir_path = Path(video_dir)
    video_files = sorted(video_dir_path.glob(video_pattern))

    if not video_files:
        print(f"在 {video_dir} 中没有找到匹配 {video_pattern} 的视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件")

    # 检查已处理的视频
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    processed_videos = set()
    if resume:
        for json_file in output_dir_path.glob("*.json"):
            processed_videos.add(json_file.stem)
        print(f"已处理 {len(processed_videos)} 个视频，将跳过它们")

    # 过滤待处理的视频
    videos_to_process = [
        v for v in video_files
        if v.stem not in processed_videos
    ]

    if not videos_to_process:
        print("所有视频都已处理完成！")
        return

    print(f"待处理视频数量: {len(videos_to_process)}")

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

    # 批量处理
    results = []
    try:
        for video_path in tqdm(videos_to_process, desc="批量处理视频"):
            video_name = video_path.stem
            video_filename = video_path.name
            output_json_path = output_dir_path / f"{video_name}.json"

            # 获取该视频的目标物体
            if config_json:
                if video_filename in video_to_prompt:
                    current_prompt = video_to_prompt[video_filename]
                else:
                    print(f"⚠ 警告: 配置文件中未找到 {video_filename}，跳过")
                    continue
            else:
                current_prompt = prompt_text

            print(f"\n处理视频: {video_name} | 目标: {current_prompt}")

            result = process_single_video(
                predictor=predictor,
                video_path=str(video_path),
                prompt_text=current_prompt,
                output_json_path=str(output_json_path),
                score_threshold=score_threshold,
            )

            results.append(result)

            # 打印处理结果
            if result["success"]:
                print(
                    f"✓ {video_name}: {result['frames_with_object']}/{result['total_frames']} 帧"
                )
            else:
                print(f"✗ {video_name}: 处理失败 - {result['error']}")

    finally:
        # 清理资源
        predictor.shutdown()

    # 保存汇总统计
    summary = {
        "video_dir": video_dir,
        "prompt_text": prompt_text,
        "total_videos": len(videos_to_process),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
    }

    summary_path = output_dir_path / "batch_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n批处理完成！")
    print(f"成功: {summary['successful']} / {len(videos_to_process)}")
    print(f"失败: {summary['failed']} / {len(videos_to_process)}")
    print(f"汇总统计已保存到: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="批量处理视频进行目标物体预分割"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="视频文件夹路径，如 /data2/xc/hf_cache/cambrians_vsr/10mins",
    )
    parser.add_argument(
        "--prompt_text",
        type=str,
        default=None,
        help="文本提示，描述要分割的目标物体（如果不使用config_json则必需）",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default=None,
        help="JSON配置文件路径，包含视频名称到目标物体的映射",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出文件夹路径",
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
        help="分数阈值（默认: 0.5）",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="不跳过已处理的视频，重新处理所有视频",
    )
    parser.add_argument(
        "--video_pattern",
        type=str,
        default="*.mp4",
        help="视频文件匹配模式（默认: *.mp4）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="本地模型权重路径（如 checkpoints/sam3.pt），如果不提供则从HuggingFace下载",
    )

    args = parser.parse_args()

    # 验证参数：必须提供 prompt_text 或 config_json 之一
    if not args.prompt_text and not args.config_json:
        parser.error("必须提供 --prompt_text 或 --config_json 参数之一")

    gpus_to_use = None
    if args.gpus is not None:
        gpus_to_use = [int(gpu_id) for gpu_id in args.gpus.split(",")]

    batch_process_videos(
        video_dir=args.video_dir,
        prompt_text=args.prompt_text,
        output_dir=args.output_dir,
        gpus_to_use=gpus_to_use,
        score_threshold=args.score_threshold,
        resume=not args.no_resume,
        video_pattern=args.video_pattern,
        config_json=args.config_json,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
