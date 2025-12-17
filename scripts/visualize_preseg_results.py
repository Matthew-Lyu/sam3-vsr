#!/usr/bin/env python
"""
可视化预分割结果 - 读取JSON文件并可视化检测结果

功能：
- 读取预分割JSON结果
- 在视频帧上绘制边界框
- 生成可视化图像或视频
- 生成统计报告
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm


def load_video_frames(video_path, max_frames=None):
    """
    加载视频帧

    Args:
        video_path: 视频文件路径（.mp4）
        max_frames: 最多加载的帧数（None表示加载所有）

    Returns:
        frames: 帧列表
    """
    if video_path.endswith('.mp4'):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1

            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        return frames
    else:
        # JPEG文件夹
        frame_files = sorted(Path(video_path).glob("*.jpg"))
        if max_frames:
            frame_files = frame_files[:max_frames]

        frames = []
        for frame_file in frame_files:
            img = cv2.imread(str(frame_file))
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        return frames


def visualize_frame_with_detections(frame, frame_info, show_labels=True):
    """
    在帧上绘制检测结果

    Args:
        frame: 图像帧（numpy array）
        frame_info: 该帧的标注信息
        show_labels: 是否显示标签

    Returns:
        vis_frame: 绘制了检测框的帧
    """
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(frame)

    if frame_info and 'objects' in frame_info:
        for obj in frame_info['objects']:
            bbox = obj['bbox']  # [x, y, w, h]
            score = obj['score']
            obj_id = obj['obj_id']

            # 绘制边界框
            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

            # 添加标签
            if show_labels:
                label = f"ID:{obj_id} {score:.2f}"
                ax.text(
                    bbox[0], bbox[1] - 10,
                    label,
                    color='white',
                    fontsize=10,
                    bbox=dict(facecolor='red', alpha=0.7, pad=2)
                )

    ax.axis('off')
    plt.tight_layout()

    # 转换为numpy array
    fig.canvas.draw()
    vis_frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    vis_frame = vis_frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    vis_frame = vis_frame[:, :, :3]  # 去掉alpha通道，只保留RGB
    plt.close(fig)

    return vis_frame


def generate_statistics(json_path):
    """
    生成统计报告

    Args:
        json_path: JSON文件路径

    Returns:
        stats: 统计信息字典
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    total_frames = data['total_frames']
    frames_with_object = data['frames_with_object']
    frame_annotations = data['frame_annotations']

    # 统计信息
    stats = {
        'video_name': data['video_name'],
        'prompt_text': data['prompt_text'],
        'total_frames': total_frames,
        'frames_with_object': len(frames_with_object),
        'detection_rate': len(frames_with_object) / total_frames if total_frames > 0 else 0,
        'total_objects': 0,
        'avg_objects_per_frame': 0,
        'max_objects_in_frame': 0,
        'unique_object_ids': set(),
    }

    # 统计每帧的对象数
    objects_per_frame = []
    for frame_idx_str, frame_info in frame_annotations.items():
        num_objects = len(frame_info['objects'])
        objects_per_frame.append(num_objects)
        stats['total_objects'] += num_objects
        stats['max_objects_in_frame'] = max(stats['max_objects_in_frame'], num_objects)

        # 收集唯一的对象ID
        for obj in frame_info['objects']:
            stats['unique_object_ids'].add(obj['obj_id'])

    if objects_per_frame:
        stats['avg_objects_per_frame'] = np.mean(objects_per_frame)

    stats['num_unique_objects'] = len(stats['unique_object_ids'])
    stats['unique_object_ids'] = sorted(list(stats['unique_object_ids']))

    return stats


def visualize_results(
    json_path,
    video_path=None,
    output_dir=None,
    num_samples=10,
    save_video=False,
):
    """
    可视化预分割结果

    Args:
        json_path: JSON结果文件路径
        video_path: 原视频路径（如果不提供，将只显示统计信息）
        output_dir: 输出目录（用于保存可视化图像）
        num_samples: 采样多少帧进行可视化
        save_video: 是否保存为视频文件
    """
    # 加载JSON结果
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 生成统计报告
    print("\n=== 统计报告 ===")
    stats = generate_statistics(json_path)
    print(f"视频名称: {stats['video_name']}")
    print(f"文本提示: {stats['prompt_text']}")
    print(f"总帧数: {stats['total_frames']}")
    print(f"检测到目标的帧数: {stats['frames_with_object']}")
    print(f"检测率: {stats['detection_rate'] * 100:.2f}%")
    print(f"总对象数: {stats['total_objects']}")
    print(f"唯一对象数: {stats['num_unique_objects']}")
    print(f"平均每帧对象数: {stats['avg_objects_per_frame']:.2f}")
    print(f"单帧最多对象数: {stats['max_objects_in_frame']}")
    print(f"对象ID列表: {stats['unique_object_ids']}")

    # 如果没有提供视频路径，只显示统计信息
    if not video_path:
        print("\n提示: 提供 --video_path 参数可以生成可视化图像")
        return

    # 加载视频帧
    print(f"\n正在加载视频: {video_path}")
    frames = load_video_frames(video_path)

    if not frames:
        print("错误: 无法加载视频帧")
        return

    print(f"加载了 {len(frames)} 帧")

    # 选择要可视化的帧
    frames_with_object = data['frames_with_object']
    if not frames_with_object:
        print("警告: 没有检测到任何目标")
        return

    # 均匀采样
    sample_indices = np.linspace(0, len(frames_with_object) - 1, min(num_samples, len(frames_with_object)), dtype=int)
    sampled_frame_indices = [frames_with_object[i] for i in sample_indices]

    print(f"\n正在可视化 {len(sampled_frame_indices)} 帧...")

    # 创建输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    # 可视化每一帧
    vis_frames = []
    for frame_idx in tqdm(sampled_frame_indices, desc="生成可视化"):
        if frame_idx >= len(frames):
            continue

        frame = frames[frame_idx]
        frame_info = data['frame_annotations'].get(str(frame_idx))

        vis_frame = visualize_frame_with_detections(frame, frame_info)
        vis_frames.append(vis_frame)

        # 保存图像
        if output_dir:
            output_file = output_path / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_file), cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

    if output_dir:
        print(f"\n可视化图像已保存到: {output_dir}")

    # 保存为视频
    if save_video and output_dir:
        output_video_path = output_path / f"{stats['video_name']}_vis.mp4"
        print(f"\n正在保存视频: {output_video_path}")

        if vis_frames:
            h, w = vis_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_video_path), fourcc, 2.0, (w, h))

            for vis_frame in vis_frames:
                out.write(cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))

            out.release()
            print(f"视频已保存: {output_video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="可视化视频预分割结果"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="JSON结果文件路径",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="原视频路径（可选，用于生成可视化）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（可选，用于保存可视化图像）",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="采样多少帧进行可视化（默认: 10）",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="是否保存为视频文件",
    )

    args = parser.parse_args()

    visualize_results(
        json_path=args.json_path,
        video_path=args.video_path,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        save_video=args.save_video,
    )


if __name__ == "__main__":
    main()
