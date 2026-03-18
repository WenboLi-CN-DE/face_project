#!/bin/bash
# SSH 公钥配置脚本 - 帮助你配置服务器的 SSH 密钥访问

PEM_FILE="/home/wenbo/personal_project/face_project/data/KeyPair-v2.pem"
USER="deploy"
HOST="159.138.228.40"

echo "=== SSH 公钥配置指南 ==="
echo ""

# 1. 检查本地 PEM 文件
echo "步骤 1: 检查本地 PEM 文件"
echo "----------------------------------------"
if [ -f "$PEM_FILE" ]; then
    PERM=$(stat -c '%a' $PEM_FILE)
    echo "✓ PEM 文件存在：$PEM_FILE"
    echo "  当前权限：$PERM"
    
    if [ "$PERM" != "600" ]; then
        echo "  ⚠️  权限不正确，正在修复..."
        chmod 600 $PEM_FILE
        echo "  ✓ 权限已设置为 600"
    else
        echo "  ✓ 权限正确 (600)"
    fi
else
    echo "❌ PEM 文件不存在：$PEM_FILE"
    exit 1
fi
echo ""

# 2. 提取公钥
echo "步骤 2: 提取公钥"
echo "----------------------------------------"
PUB_KEY=$(ssh-keygen -y -f $PEM_FILE 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✓ 公钥提取成功:"
    echo "  $PUB_KEY" | head -c 100
    echo "..."
else
    echo "❌ 无法提取公钥，PEM 文件可能已损坏"
    exit 1
fi
echo ""

# 3. 服务器端配置说明
echo "步骤 3: 配置服务器（需要手动操作）"
echo "----------------------------------------"
echo "由于当前无法通过密钥登录，请按以下步骤操作:"
echo ""
echo "3.1 首先通过密码或其他方式登录服务器:"
echo "    ssh $USER@$HOST"
echo ""
echo "3.2 登录服务器后，执行以下命令:"
echo "    # 创建 .ssh 目录"
echo "    mkdir -p ~/.ssh"
echo "    chmod 700 ~/.ssh"
echo ""
echo "    # 创建或编辑 authorized_keys"
echo "    touch ~/.ssh/authorized_keys"
echo "    chmod 600 ~/.ssh/authorized_keys"
echo ""
echo "    # 添加公钥（将下面的公钥内容复制粘贴进去）"
echo "    echo '$PUB_KEY' >> ~/.ssh/authorized_keys"
echo ""
echo "    # 或者使用 authorized_keys 编辑器"
echo "    vi ~/.ssh/authorized_keys"
echo "    # 按 'i' 进入插入模式"
echo "    # 粘贴公钥（右键或 Shift+Insert）"
echo "    # 按 ESC，然后输入 :wq 保存退出"
echo ""
echo "    # 确保权限正确"
echo "    chmod 700 ~/.ssh"
echo "    chmod 600 ~/.ssh/authorized_keys"
echo ""
echo "3.3 退出服务器"
echo "    exit"
echo ""
echo "3.4 测试密钥登录"
echo "    ssh -i $PEM_FILE -o IdentitiesOnly=yes $USER@$HOST"
echo ""

# 4. 一键配置（如果已有其他密钥可以登录）
echo "步骤 4: 一键配置（可选）"
echo "----------------------------------------"
echo "如果你已经有其他密钥可以登录服务器，可以使用以下命令一键配置:"
echo ""
echo "    ssh-keygen -y -f $PEM_FILE | \\"
echo "      ssh -o IdentitiesOnly=yes $USER@$HOST \\"
echo "      'mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys'"
echo ""

echo "=== 完成 ==="
