#!/bin/bash
# 四卡GPU并行运行启动脚本

echo "=================================="
echo "  参数敏感性分析 - 四卡GPU并行运行"
echo "=================================="
echo ""

# 检查GPU配置
echo "1. 检查GPU配置..."
python3 check_gpu_setup.py
echo ""

# 提示信息
echo "2. 运行主程序..."
echo "   启动四卡并行处理..."
echo ""

# 运行主程序
python3 sml4_sensitivity.py

echo ""
echo "=================================="
echo "✓ 并行处理完成！"
echo "=================================="
echo "输出位置:"
echo "  - 热力图: ./sensitivity_results/"
echo "  - P和A分解: ./sensitivity/"
echo ""
echo "查看监控命令:"
echo "  watch -n 1 nvidia-smi"
echo ""
