#!/usr/bin/env python
"""
计算SAM3检测率

规则：
- 如果检测帧数 >= 预期帧数，算成功
- 否则算失败
"""

import argparse
import json
from pathlib import Path


def calculate_detection_rate(ground_truth_file, batch_summary_file):
    """
    计算检测率

    Args:
        ground_truth_file: ground truth JSON文件路径
        batch_summary_file: batch_summary JSON文件路径
    """
    # 读取ground truth
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # 读取检测结果
    with open(batch_summary_file, 'r', encoding='utf-8') as f:
        batch_summary = json.load(f)

    # 构建检测结果映射 (video_name -> detected_frames)
    detection_map = {}
    for result in batch_summary.get('results', []):
        if result.get('success'):
            video_name = result['video_name'] + '.mp4'
            detection_map[video_name] = result.get('frames_with_object', 0)

    # 对比结果
    total_videos = len(ground_truth)
    success_count = 0
    fail_count = 0
    missing_count = 0

    success_videos = []
    fail_videos = []
    missing_videos = []

    for video_name, expected_count in sorted(ground_truth.items()):
        if video_name not in detection_map:
            missing_count += 1
            missing_videos.append({
                'video': video_name,
                'expected': expected_count,
                'detected': 0,
                'reason': '未找到检测结果'
            })
            continue

        detected_count = detection_map[video_name]

        # 检测数量 >= 预期数量，算成功
        if detected_count >= expected_count:
            success_count += 1
            success_videos.append({
                'video': video_name,
                'expected': expected_count,
                'detected': detected_count
            })
        else:
            fail_count += 1
            fail_videos.append({
                'video': video_name,
                'expected': expected_count,
                'detected': detected_count,
                'diff': detected_count - expected_count
            })

    # 计算检测率
    detection_rate = success_count / total_videos if total_videos > 0 else 0

    # 打印报告
    print("\n" + "=" * 80)
    print("SAM3 检测率统计报告")
    print("=" * 80)

    print(f"\n【总体统计】")
    print(f"  总视频数: {total_videos}")
    print(f"  成功检测: {success_count} ({success_count/total_videos*100:.1f}%)")
    print(f"  检测不足: {fail_count} ({fail_count/total_videos*100:.1f}%)")
    print(f"  缺失结果: {missing_count} ({missing_count/total_videos*100:.1f}%)")
    print(f"  检测率: {detection_rate*100:.2f}%")

    # 显示失败的视频
    if fail_videos:
        print(f"\n【检测不足的视频】({len(fail_videos)} 个)")
        for item in fail_videos:
            print(f"  {item['video']}: 预期 {item['expected']}, 检测 {item['detected']}, 差 {item['diff']}")

    # 显示缺失结果的视频
    if missing_videos:
        print(f"\n【缺失检测结果的视频】({len(missing_videos)} 个)")
        for item in missing_videos:
            print(f"  {item['video']}: {item['reason']}")

    # 统计检测帧数分布
    print(f"\n【检测帧数分布】")
    frame_distribution = {}
    for video_name, detected_count in detection_map.items():
        if detected_count not in frame_distribution:
            frame_distribution[detected_count] = 0
        frame_distribution[detected_count] += 1

    for frame_count in sorted(frame_distribution.keys()):
        count = frame_distribution[frame_count]
        print(f"  检测到 {frame_count} 帧: {count} 个视频")

    print("\n" + "=" * 80)

    # 返回统计结果
    return {
        'total_videos': total_videos,
        'success_count': success_count,
        'fail_count': fail_count,
        'missing_count': missing_count,
        'detection_rate': round(detection_rate, 4),
        'success_videos': success_videos,
        'fail_videos': fail_videos,
        'missing_videos': missing_videos,
        'frame_distribution': frame_distribution
    }


def main():
    parser = argparse.ArgumentParser(
        description="计算SAM3检测率"
    )
    parser.add_argument(
        '--ground_truth',
        type=str,
        default='ground_truth.json',
        help='ground truth JSON文件路径（默认: ground_truth.json）'
    )
    parser.add_argument(
        '--batch_summary',
        type=str,
        default='batch_summary.json',
        help='batch_summary JSON文件路径（默认: batch_summary.json）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='detection_rate_report.json',
        help='输出报告JSON文件路径（默认: detection_rate_report.json）'
    )

    args = parser.parse_args()

    # 计算检测率
    report = calculate_detection_rate(args.ground_truth, args.batch_summary)

    # 保存报告
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n详细报告已保存到: {output_path}")


if __name__ == '__main__':
    main()
