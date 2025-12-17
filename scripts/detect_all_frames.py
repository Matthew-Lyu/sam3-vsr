#!/usr/bin/env python
"""
在视频的所有帧（或每N帧）上添加text prompt进行检测
不使用tracking，每一帧独立检测
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_predictor


def detect_in_all_frames(
    video_path: str,
    prompt_text: str,
    output_json_path: str,
    checkpoint_path: str,
    gpus_to_use: list,
    score_threshold: float = 0.5,
    frame_stride: int = 1,  # 每隔多少帧检测一次，1表示每帧都检测
):
    """
    在所有帧上独立检测物体（不使用tracking）

    Args:
        video_path: 视频路径
        prompt_text: 文本提示
        output_json_path: 输出JSON路径
        checkpoint_path: 模型权重路径
        gpus_to_use: GPU列表
        score_threshold: 分数阈值
        frame_stride: 帧间隔，1表示每帧都检测
    """

    print(f"=" * 80)
    print(f"视频: {video_path}")
    print(f"目标: {prompt_text}")
    print(f"帧间隔: {frame_stride} (1=检测所有帧)")
    print(f"分数阈值: {score_threshold}")
    print(f"=" * 80)

    # 初始化预测器
    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        checkpoint_path=checkpoint_path
    )

    try:
        # 启动会话
        print("\n正在加载视频...")
        response = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        session_id = response["session_id"]

        # 先在frame 0添加prompt，然后propagate获取总帧数
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

        # 在每一帧（或每N帧）上检测
        frames_to_check = list(range(0, frame_count, frame_stride))
        print(f"将检测 {len(frames_to_check)} 帧\n")

        frames_with_object = []
        frame_annotations = {}

        for frame_idx in tqdm(frames_to_check, desc="检测中"):
            # 重置会话以清除之前的检测
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

            outputs = response["outputs"]

            # 检查是否有检测结果
            if "out_probs" not in outputs or len(outputs["out_probs"]) == 0:
                continue

            # 过滤低分数的检测
            valid_indices = [
                i for i, score in enumerate(outputs["out_probs"])
                if score >= score_threshold
            ]

            if not valid_indices:
                continue

            # 记录这一帧
            frames_with_object.append(frame_idx)

            # 保存检测结果
            objects = []
            for idx in valid_indices:
                obj_info = {
                    "obj_id": int(outputs["out_obj_ids"][idx]) if "out_obj_ids" in outputs else idx,
                    "score": float(outputs["out_probs"][idx]),
                }

                if "out_boxes_xywh" in outputs and outputs["out_boxes_xywh"] is not None:
                    box = outputs["out_boxes_xywh"][idx]
                    obj_info["bbox"] = [
                        float(box[0]), float(box[1]),
                        float(box[2]), float(box[3]),
                    ]

                if "out_binary_masks" in outputs and outputs["out_binary_masks"] is not None:
                    mask = outputs["out_binary_masks"][idx]
                    obj_info["mask_area"] = int(mask.sum())

                objects.append(obj_info)

            frame_annotations[str(frame_idx)] = {
                "frame_idx": frame_idx,
                "objects": objects,
            }

        # 保存结果
        result = {
            "video_path": video_path,
            "video_name": Path(video_path).stem,
            "prompt_text": prompt_text,
            "score_threshold": score_threshold,
            "frame_stride": frame_stride,
            "total_frames": frame_count,
            "frames_checked": len(frames_to_check),
            "num_frames_with_object": len(frames_with_object),
            "frames_with_object": sorted(frames_with_object),
            "frame_annotations": frame_annotations,
        }

        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n" + "=" * 80)
        print(f"结果已保存: {output_json_path}")
        print(f"总帧数: {frame_count}")
        print(f"检测帧数: {len(frames_to_check)}")
        print(f"检测到目标的帧数: {len(frames_with_object)}")
        if len(frames_to_check) > 0:
            print(f"检测率: {len(frames_with_object) / len(frames_to_check) * 100:.2f}%")
        print(f"=" * 80)

        # 关闭会话
        predictor.handle_request(
            request=dict(
                type="close_session",
                session_id=session_id,
            )
        )

    finally:
        predictor.shutdown()


def main():
    parser = argparse.ArgumentParser(description="在所有帧上独立检测物体")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--prompt_text", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=10,
        help="每隔多少帧检测一次，1表示每帧都检测（慢但准确），10表示每10帧检测一次（快但可能遗漏）"
    )

    args = parser.parse_args()

    gpus_to_use = [int(gpu_id) for gpu_id in args.gpus.split(",")]

    detect_in_all_frames(
        video_path=args.video_path,
        prompt_text=args.prompt_text,
        output_json_path=args.output,
        checkpoint_path=args.checkpoint,
        gpus_to_use=gpus_to_use,
        score_threshold=args.score_threshold,
        frame_stride=args.frame_stride,
    )


if __name__ == "__main__":
    main()
