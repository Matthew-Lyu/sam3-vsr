#!/usr/bin/env python
"""
从VSR数据集的prompt文件中提取ground truth信息

功能：
- 解析all_10mins_prompts.txt文件
- 提取每个视频的目标物体和应该出现的次数（地点数量）
- 生成ground_truth.json文件，用于对比检测结果
"""

import argparse
import json
import re
from pathlib import Path


def parse_prompts_file(prompt_file_path: str):
    """
    解析prompt文件，提取每个视频的信息

    Args:
        prompt_file_path: prompt文件路径

    Returns:
        dict: 视频名称 -> {target_object, expected_count, locations}
    """
    with open(prompt_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按分隔线分割每个视频的问题
    video_blocks = content.split('----------------------------------------')

    ground_truth = {}

    for block in video_blocks:
        block = block.strip()
        if not block:
            continue

        # 提取视频路径和索引
        video_match = re.search(r'Video Index: (\d+) \| Path: (.+?)\.mp4', block)
        if not video_match:
            continue

        video_index = video_match.group(1)
        video_path = video_match.group(2)
        video_name = Path(video_path).name + '.mp4'

        # 提取目标物体
        # 格式: "Which of the following correctly represents the order in which the XXX appeared in the video?"
        object_match = re.search(
            r'order in which the (.+?) appeared in the video',
            block,
            re.IGNORECASE
        )

        if not object_match:
            print(f"警告: 无法从视频 {video_name} 中提取目标物体")
            continue

        target_object = object_match.group(1).strip()

        # 提取所有选项中的地点
        # 每个选项格式: "A. Location1, Location2, Location3, Location4"
        options = re.findall(r'[A-D]\.\s+(.+)', block)

        if not options:
            print(f"警告: 无法从视频 {video_name} 中提取选项")
            continue

        # 解析第一个选项来获取地点数量（所有选项地点数量应该相同）
        locations = [loc.strip() for loc in options[0].split(',')]
        expected_count = len(locations)

        # 收集所有可能的地点（从所有选项中）
        all_locations = set()
        for option in options:
            locs = [loc.strip() for loc in option.split(',')]
            all_locations.update(locs)

        ground_truth[video_name] = expected_count

    return ground_truth


def main():
    parser = argparse.ArgumentParser(
        description="从VSR数据集prompt文件中提取ground truth信息"
    )
    parser.add_argument(
        '--prompt_file',
        type=str,
        default='all_10mins_prompts.txt',
        help='prompt文件路径（默认: all_10mins_prompts.txt）'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ground_truth.json',
        help='输出JSON文件路径（默认: ground_truth.json）'
    )

    args = parser.parse_args()

    print(f"正在解析prompt文件: {args.prompt_file}")
    ground_truth = parse_prompts_file(args.prompt_file)

    print(f"\n成功提取 {len(ground_truth)} 个视频的ground truth信息")

    # 统计信息
    total_expected = sum(ground_truth.values())
    print(f"总共应该检测到的出现次数: {total_expected}")

    # 保存JSON文件
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"\nGround truth信息已保存到: {output_path}")

    # 打印几个示例
    print("\n示例（前5个视频）:")
    for video_name in sorted(ground_truth.keys())[:5]:
        count = ground_truth[video_name]
        print(f"  {video_name}: {count}")


if __name__ == '__main__':
    main()
