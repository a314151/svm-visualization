# SVM 可视化教学系统

交互式学习支持向量机（Support Vector Machine）的原理与计算过程。

## 🌐 在线访问

| 平台 | 地址 |
|------|------|
| **GitHub** | https://github.com/a314151/svm-visualization |
| **Vercel** | https://svm-visualization.vercel.app |
| **Cloudflare Pages** | https://svm-visualization.pages.dev |

## ✨ 功能特性

- 📊 **交互式画布** - 点击添加数据点，实时可视化
- 🔧 **多种核函数** - 支持线性核和高斯核(RBF)
- 📈 **SMO算法** - 实现序列最小优化算法求解对偶问题
- 📋 **计算过程详解** - 核矩阵、Alpha值、决策边界可视化
- 📚 **公式推导** - 展示SVM的数学原理

## 🚀 快速开始

### 本地开发

```bash
# 克隆仓库
git clone https://github.com/a314151/svm-visualization.git
cd svm-visualization

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

### 使用说明

1. **添加数据点** - 选择"正类"或"负类"标签，在画布上点击添加
2. **选择核函数** - 线性核适用于线性可分数据，高斯核适用于非线性数据
3. **调整参数** - Gamma控制高斯核宽度，C控制对误分类的惩罚
4. **训练模型** - 点击"训练SVM"按钮，查看计算结果
5. **分析结果** - 查看核矩阵、Alpha值和决策边界

## 📖 技术细节

### SMO算法

本系统实现了完整的SMO（Sequential Minimal Optimization）算法：

1. **对偶问题求解** - 将原始优化问题转化为对偶问题
2. **KKT条件检验** - 检查样本是否违反KKT条件
3. **启发式变量选择** - 选择使|E1-E2|最大的变量对
4. **边界约束** - 确保alpha在[0, C]范围内

### 核函数

- **线性核**: K(xᵢ, xⱼ) = xᵢ · xⱼ
- **高斯核**: K(xᵢ, xⱼ) = exp(-γ||xᵢ - xⱼ||²)

## 🛠️ 技术栈

- **框架**: Next.js 16
- **语言**: TypeScript
- **样式**: Tailwind CSS
- **UI组件**: shadcn/ui
- **算法**: 原生JavaScript实现SMO

## 📝 更新日志

### v1.0.1 (2026-03-16)
- 修复SMO算法中b和errors更新的值传递问题
- 使用状态对象解决JavaScript值传递bug

### v1.0.0 (2026-03-16)
- 初始版本发布
- 支持线性核和高斯核
- 实现SMO算法
- 交互式可视化界面

## 📄 许可证

MIT License

## 🔗 相关链接

- [GitHub仓库](https://github.com/a314151/svm-visualization)
- [Vercel部署](https://svm-visualization.vercel.app)
- [Cloudflare Pages部署](https://svm-visualization.pages.dev)
