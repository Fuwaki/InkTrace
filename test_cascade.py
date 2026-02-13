import torch
import torch.nn as nn
from encoder import StrokeEncoder
from decoder import UniversalDecoder
from dense_heads import DenseHeads


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_num(num):
    """格式化数字显示"""
    if num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return str(num)


def print_separator(title=""):
    """打印分隔符"""
    width = 60
    if title:
        title = f" {title} "
        left = (width - len(title)) // 2
        right = width - len(title) - left
        print("=" * left + title + "=" * right)
    else:
        print("=" * width)


def test_cascade():
    """测试 Encoder -> Decoder -> DenseHeads 级联"""
    print_separator("级联测试")

    # ========== 测试配置 ==========
    batch_size = 2
    embed_dim = 64
    num_heads = 4
    num_layers = 2

    # ========== 创建模块 ==========
    print("\n[1] 创建模块...")
    encoder = StrokeEncoder(
        in_channels=1,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=0.1,
    ).eval()

    decoder = UniversalDecoder(
        embed_dim=embed_dim,
    ).eval()

    # DenseHeads 已集成在 Decoder 中，这里单独测试
    dense_heads = DenseHeads(
        in_channels=16,
        head_channels=48,
    ).eval()

    # ========== 统计参数量 ==========
    print("\n[2] 参数量统计:")
    enc_total, enc_train = count_parameters(encoder)
    dec_total, dec_train = count_parameters(decoder)
    heads_total, heads_train = count_parameters(dense_heads)

    print(f"  Encoder:  Total: {format_num(enc_total):>8}  Trainable: {format_num(enc_train):>8}")
    print(f"  Decoder:  Total: {format_num(dec_total):>8}  Trainable: {format_num(dec_train):>8}")
    print(f"  DenseHeads: Total: {format_num(heads_total):>8}  Trainable: {format_num(heads_train):>8}")
    print(f"  {'─' * 44}")
    print(f"  Combined: Total: {format_num(enc_total + dec_total):>8}  Trainable: {format_num(enc_train + dec_train):>8}")

    # ========== 创建测试输入 ==========
    print("\n[3] 创建测试输入...")
    x = torch.randn(batch_size, 1, 64, 64)
    print(f"  Input shape: {tuple(x.shape)}")

    # ========== 测试 Encoder ==========
    print("\n[4] 测试 Encoder:")
    with torch.no_grad():
        # 测试只返回 embeddings
        embeddings = encoder(x, return_interm_layers=False)
        print(f"  Embeddings shape: {tuple(embeddings.shape)}")
        assert embeddings.shape == (batch_size, 16, embed_dim), \
            f"Embeddings shape mismatch: {embeddings.shape} != ({batch_size}, 16, {embed_dim})"

        # 测试返回中间层特征
        features, embeddings_full = encoder(x, return_interm_layers=True)
        f1, f2, f3, f4 = features
        print(f"  F1 (Stage 1):    {tuple(f1.shape)}  - 期望: ({batch_size}, 16, 32, 32)")
        print(f"  F2 (Stage 2):    {tuple(f2.shape)}  - 期望: ({batch_size}, 32, 16, 16)")
        print(f"  F3 (Stage 3):    {tuple(f3.shape)}  - 期望: ({batch_size}, 64, 8, 8)")
        print(f"  F4 (Bottleneck): {tuple(f4.shape)}  - 期望: ({batch_size}, {embed_dim}, 4, 4)")
        print(f"  Embeddings:      {tuple(embeddings_full.shape)}  - 期望: ({batch_size}, 16, {embed_dim})")

        # 验证形状
        assert f1.shape == (batch_size, 16, 32, 32), f"F1 shape mismatch: {f1.shape}"
        assert f2.shape == (batch_size, 32, 16, 16), f"F2 shape mismatch: {f2.shape}"
        assert f3.shape == (batch_size, 64, 8, 8), f"F3 shape mismatch: {f3.shape}"
        assert f4.shape == (batch_size, embed_dim, 4, 4), f"F4 shape mismatch: {f4.shape}"

    # ========== 测试 Decoder (带 skip) ==========
    print("\n[5] 测试 Decoder (带 skip):")
    with torch.no_grad():
        decoder.set_skip_mode('learnable')
        outputs_skip = decoder(features, use_skips=True)

        print(f"  Skeleton:  {tuple(outputs_skip['skeleton'].shape)}  - 期望: ({batch_size}, 1, 64, 64)")
        print(f"  Tangent:   {tuple(outputs_skip['tangent'].shape)}  - 期望: ({batch_size}, 2, 64, 64)")
        print(f"  Width:     {tuple(outputs_skip['width'].shape)}  - 期望: ({batch_size}, 1, 64, 64)")
        print(f"  Offset:    {tuple(outputs_skip['offset'].shape)}  - 期望: ({batch_size}, 2, 64, 64)")
        print(f"  Keypoints: {tuple(outputs_skip['keypoints'].shape)}  - 期望: ({batch_size}, 2, 64, 64)")

        # 验证形状
        assert outputs_skip['skeleton'].shape == (batch_size, 1, 64, 64)
        assert outputs_skip['tangent'].shape == (batch_size, 2, 64, 64)
        assert outputs_skip['width'].shape == (batch_size, 1, 64, 64)
        assert outputs_skip['offset'].shape == (batch_size, 2, 64, 64)
        assert outputs_skip['keypoints'].shape == (batch_size, 2, 64, 64)

    # ========== 测试 Decoder (无 skip) ==========
    print("\n[6] 测试 Decoder (无 skip):")
    with torch.no_grad():
        decoder.set_skip_mode('frozen')
        outputs_no_skip = decoder(features, use_skips=False)

        print(f"  Skeleton:  {tuple(outputs_no_skip['skeleton'].shape)}")
        print(f"  Tangent:   {tuple(outputs_no_skip['tangent'].shape)}")
        print(f"  Width:     {tuple(outputs_no_skip['width'].shape)}")
        print(f"  Offset:    {tuple(outputs_no_skip['offset'].shape)}")
        print(f"  Keypoints: {tuple(outputs_no_skip['keypoints'].shape)}")

    # ========== 测试 DenseHeads 单独使用 ==========
    print("\n[7] 测试 DenseHeads (单独使用):")
    decoder_output_feat = torch.randn(batch_size, 16, 64, 64)
    with torch.no_grad():
        heads_output = dense_heads(decoder_output_feat)

        print(f"  Skeleton:  {tuple(heads_output['skeleton'].shape)}")
        print(f"  Tangent:   {tuple(heads_output['tangent'].shape)}")
        print(f"  Width:     {tuple(heads_output['width'].shape)}")
        print(f"  Offset:    {tuple(heads_output['offset'].shape)}")
        print(f"  Keypoints: {tuple(heads_output['keypoints'].shape)}")

    # ========== 端到端测试 ==========
    print("\n[8] 端到端测试 (Encoder -> Decoder):")
    with torch.no_grad():
        features, _ = encoder(x, return_interm_layers=True)
        outputs_e2e = decoder(features, use_skips=True)

        print(f"  Input:      {tuple(x.shape)}")
        print(f"  Output:     {tuple(outputs_e2e['skeleton'].shape)}")

    # ========== 验证数值范围 ==========
    print("\n[9] 验证输出数值范围:")
    print(f"  Skeleton:   [{outputs_skip['skeleton'].min():.4f}, {outputs_skip['skeleton'].max():.4f}]  (期望: [0, 1])")
    print(f"  Tangent:    [{outputs_skip['tangent'].min():.4f}, {outputs_skip['tangent'].max():.4f}]  (期望: [-1, 1])")
    print(f"  Width:      [{outputs_skip['width'].min():.4f}, {outputs_skip['width'].max():.4f}]  (期望: >=0)")
    print(f"  Offset:     [{outputs_skip['offset'].min():.4f}, {outputs_skip['offset'].max():.4f}]  (期望: [-0.5, 0.5])")
    print(f"  Keypoints:  [{outputs_skip['keypoints'].min():.4f}, {outputs_skip['keypoints'].max():.4f}]  (期望: [0, 1])")

    # 验证数值范围
    assert outputs_skip['skeleton'].min() >= 0 and outputs_skip['skeleton'].max() <= 1
    assert outputs_skip['tangent'].min() >= -1 and outputs_skip['tangent'].max() <= 1
    assert outputs_skip['width'].min() >= 0
    assert outputs_skip['offset'].min() >= -0.5 and outputs_skip['offset'].max() <= 0.5
    assert outputs_skip['keypoints'].min() >= 0 and outputs_skip['keypoints'].max() <= 1

    print_separator("所有测试通过!")
    print()


if __name__ == "__main__":
    test_cascade()
