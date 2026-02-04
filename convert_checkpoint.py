#!/usr/bin/env python3
"""
Checkpoint 转换脚本

将旧版 Encoder 权重 (stem 单一 Sequential) 转换为新版架构 (stem1 + stem2 拆分)。

使用方法:
    python convert_checkpoint.py --input checkpoints/phase1_epoch10.pth --output checkpoints/phase1_epoch10_v2.pth

    # 批量转换
    python convert_checkpoint.py --input-dir checkpoints --output-dir checkpoints_v2

旧架构 stem 结构:
    stem.0: Conv2d_BN (1->32, stride=2)   -> stem1.0
    stem.1: GELU                          -> stem1.1 (无权重)
    stem.2: Conv2d_BN (32->64, stride=2)  -> stem2.0
    stem.3: GELU                          -> stem2.1 (无权重)
    stem.4: Conv2d_BN (64->128, stride=1) -> stem2.2

新架构:
    stem1: Sequential[Conv2d_BN, GELU]
    stem2: Sequential[Conv2d_BN, GELU, Conv2d_BN]
"""

import argparse
import os
from pathlib import Path
import torch


def convert_encoder_state_dict(old_state_dict: dict) -> dict:
    """
    转换 Encoder state_dict 从旧架构到新架构

    Args:
        old_state_dict: 旧版 encoder 的 state_dict

    Returns:
        new_state_dict: 新版 encoder 的 state_dict
    """
    new_state_dict = {}

    for key, value in old_state_dict.items():
        if key.startswith("stem."):
            # 解析旧 key: stem.{idx}.{subkey}
            parts = key.split(".")
            idx = int(parts[1])
            suffix = ".".join(parts[2:])

            # 映射关系
            if idx == 0:
                # stem.0 -> stem1.0
                new_key = f"stem1.0.{suffix}"
            elif idx == 1:
                # stem.1 (GELU) -> stem1.1 (通常无权重，跳过)
                # GELU 没有可学习参数，但以防万一
                new_key = f"stem1.1.{suffix}"
            elif idx == 2:
                # stem.2 -> stem2.0
                new_key = f"stem2.0.{suffix}"
            elif idx == 3:
                # stem.3 (GELU) -> stem2.1 (通常无权重，跳过)
                new_key = f"stem2.1.{suffix}"
            elif idx == 4:
                # stem.4 -> stem2.2
                new_key = f"stem2.2.{suffix}"
            else:
                print(f"  警告: 未知的 stem 索引 {idx}，跳过 key: {key}")
                continue

            new_state_dict[new_key] = value
        else:
            # 其他 key 保持不变
            new_state_dict[key] = value

    return new_state_dict


def convert_checkpoint(input_path: str, output_path: str, verbose: bool = True) -> bool:
    """
    转换单个 checkpoint 文件

    Args:
        input_path: 输入 checkpoint 路径
        output_path: 输出 checkpoint 路径
        verbose: 是否打印详细信息

    Returns:
        success: 是否成功
    """
    if verbose:
        print(f"\n转换: {input_path}")

    try:
        checkpoint = torch.load(input_path, map_location="cpu")
    except Exception as e:
        print(f"  错误: 无法加载文件 - {e}")
        return False

    # 检测 checkpoint 格式
    if "encoder_state_dict" in checkpoint:
        # 标准格式
        old_encoder = checkpoint["encoder_state_dict"]
        new_encoder = convert_encoder_state_dict(old_encoder)
        checkpoint["encoder_state_dict"] = new_encoder

        if verbose:
            print(f"  ✓ 转换 encoder_state_dict")

    elif "model_state_dict" in checkpoint:
        # 完整模型格式 (encoder + decoder)
        old_model = checkpoint["model_state_dict"]

        # 分离 encoder 和其他部分
        encoder_keys = [k for k in old_model.keys() if k.startswith("encoder.")]

        if encoder_keys:
            # 提取 encoder 部分
            old_encoder = {
                k.replace("encoder.", ""): v
                for k, v in old_model.items()
                if k.startswith("encoder.")
            }
            new_encoder = convert_encoder_state_dict(old_encoder)

            # 重建 model_state_dict
            new_model = {}
            for k, v in old_model.items():
                if k.startswith("encoder."):
                    # 跳过，稍后添加转换后的
                    pass
                else:
                    new_model[k] = v

            # 添加转换后的 encoder
            for k, v in new_encoder.items():
                new_model[f"encoder.{k}"] = v

            checkpoint["model_state_dict"] = new_model

            if verbose:
                print(f"  ✓ 转换 model_state_dict 中的 encoder 部分")
        else:
            print(f"  警告: model_state_dict 中未找到 encoder 前缀的 key")

    else:
        # 可能是纯 state_dict
        if any(k.startswith("stem.") for k in checkpoint.keys()):
            checkpoint = convert_encoder_state_dict(checkpoint)
            if verbose:
                print(f"  ✓ 转换纯 state_dict")
        else:
            print(f"  跳过: 未检测到需要转换的 stem 结构")
            return False

    # 添加版本标记
    checkpoint["_converted_to_v2"] = True

    # 保存
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(checkpoint, output_path)

    if verbose:
        print(f"  ✓ 保存到: {output_path}")

    return True


def verify_conversion(checkpoint_path: str) -> bool:
    """
    验证转换后的 checkpoint 能否正常加载

    Args:
        checkpoint_path: checkpoint 路径

    Returns:
        success: 是否成功
    """
    print(f"\n验证: {checkpoint_path}")

    try:
        from encoder import StrokeEncoder
        from pixel_decoder import PixelDecoder

        # 创建模型
        encoder = StrokeEncoder(in_channels=1, embed_dim=128)
        decoder = PixelDecoder(embed_dim=128)

        # 加载 checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 尝试加载权重
        if "encoder_state_dict" in checkpoint:
            msg = encoder.load_state_dict(
                checkpoint["encoder_state_dict"], strict=False
            )
            print(
                f"  Encoder: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
            )

            if msg.missing_keys:
                print(f"    Missing: {msg.missing_keys[:5]}...")
            if msg.unexpected_keys:
                print(f"    Unexpected: {msg.unexpected_keys[:5]}...")

        if "decoder_state_dict" in checkpoint:
            msg = decoder.load_state_dict(
                checkpoint["decoder_state_dict"], strict=False
            )
            print(
                f"  Decoder: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
            )

        # 测试前向传播
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            x = torch.randn(1, 1, 64, 64)
            emb = encoder(x)
            out = decoder(emb)

        print(
            f"  ✓ 前向传播成功: input={x.shape} -> emb={emb.shape} -> output={out.shape}"
        )
        return True

    except Exception as e:
        print(f"  ✗ 验证失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="转换旧版 Encoder checkpoint 到新版架构",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换单个文件
  python convert_checkpoint.py -i checkpoints/phase1_epoch10.pth -o checkpoints/phase1_epoch10_v2.pth
  
  # 转换并验证
  python convert_checkpoint.py -i checkpoints/phase1_epoch10.pth -o checkpoints/phase1_epoch10_v2.pth --verify
  
  # 批量转换目录下所有 .pth 文件
  python convert_checkpoint.py --input-dir checkpoints --output-dir checkpoints_v2
        """,
    )

    # 单文件模式
    parser.add_argument("-i", "--input", type=str, help="输入 checkpoint 路径")
    parser.add_argument("-o", "--output", type=str, help="输出 checkpoint 路径")

    # 批量模式
    parser.add_argument("--input-dir", type=str, help="输入目录（批量转换）")
    parser.add_argument("--output-dir", type=str, help="输出目录（批量转换）")

    # 选项
    parser.add_argument("--verify", action="store_true", help="转换后验证加载")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出文件")

    args = parser.parse_args()

    # 确定模式
    if args.input and args.output:
        # 单文件模式
        if os.path.exists(args.output) and not args.overwrite:
            print(f"输出文件已存在: {args.output}")
            print("使用 --overwrite 覆盖")
            return

        success = convert_checkpoint(args.input, args.output)

        if success and args.verify:
            verify_conversion(args.output)

    elif args.input_dir and args.output_dir:
        # 批量模式
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)

        if not input_dir.exists():
            print(f"输入目录不存在: {input_dir}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)

        pth_files = list(input_dir.glob("*.pth"))
        print(f"找到 {len(pth_files)} 个 .pth 文件")

        success_count = 0
        for pth_file in pth_files:
            output_path = output_dir / pth_file.name

            if output_path.exists() and not args.overwrite:
                print(f"跳过 (已存在): {pth_file.name}")
                continue

            if convert_checkpoint(str(pth_file), str(output_path)):
                success_count += 1

                if args.verify:
                    verify_conversion(str(output_path))

        print(f"\n转换完成: {success_count}/{len(pth_files)}")

    else:
        parser.print_help()
        print("\n错误: 必须指定 --input/--output 或 --input-dir/--output-dir")


if __name__ == "__main__":
    main()
