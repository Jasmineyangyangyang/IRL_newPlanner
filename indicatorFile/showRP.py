import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体（如果需要显示中文）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_visualize(pkl_path='./indicatorFile/risk_indicator_data.pkl'):
    """
    读取风险指标数据并绘制三张图：
    1. 所有指标随 step 变化（分面子图）
    2. 多指标对比（同一坐标轴）
    3. 统计分布（箱线图和小提琴图）
    """
    
    # ==================== 读取数据 ====================
    print("=" * 60)
    print("正在读取数据...")
    
    if not os.path.exists(pkl_path):
        print(f"❌ 文件不存在: {pkl_path}")
        return None
    
    df = pd.read_pickle(pkl_path)
    
    # 数据检查
    print(f"✅ 数据加载成功！")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"\n前5行数据:\n{df.head()}")
    print(f"\n数据统计:\n{df.describe()}")
    
    # 检查缺失值
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️ 缺失值:\n{missing[missing > 0]}")
    else:
        print("\n✅ 无缺失值")
    
    # 定义指标列（排除 step）
    indicators = ['maxRP', 'maxSTLC', 'maxCTLC', 'maxCTTC', 'maxCTAD']
    
    # 确保输出目录存在
    output_dir = os.path.dirname(pkl_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== 图1: 所有指标随 step 变化（分面子图） ====================
    print("\n" + "=" * 60)
    print("正在绘制图1: 各指标随 step 变化...")
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for ax, indicator, color in zip(axes, indicators, colors):
        ax.fill_between(df['step'], df[indicator], alpha=0.3, color=color)
        ax.plot(df['step'], df[indicator], color=color, linewidth=1.5, label=indicator)
        ax.set_ylabel(indicator, fontsize=11, fontweight='bold')
        ax.tick_params(axis='y')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=9)
        
        # 添加均值线
        mean_val = df[indicator].mean()
        ax.axhline(y=mean_val, color='black', linestyle='--', alpha=0.5, 
                   label=f'mean={mean_val:.3f}')
    
    axes[-1].set_xlabel('Step', fontsize=12, fontweight='bold')
    fig.suptitle('Risk Indicators Over Steps (Individual View)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, 'fig1_individual_timeline.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图1已保存: {fig1_path}")
    plt.show()
    
    # ==================== 图2: 多指标对比（同一坐标轴） ====================
    print("\n" + "=" * 60)
    print("正在绘制图2: 多指标对比...")
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for indicator, color in zip(indicators, colors):
        ax.plot(df['step'], df[indicator], label=indicator, 
                linewidth=1.8, color=color, alpha=0.8)
    
    ax.set_xlabel('Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk Indicator Value', fontsize=12, fontweight='bold')
    ax.set_title('All Risk Indicators Comparison (Overlay)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加统计注释
    textstr = '\n'.join([
        f'{ind}: μ={df[ind].mean():.3f}, σ={df[ind].std():.3f}' 
        for ind in indicators
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, 'fig2_overlay_comparison.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图2已保存: {fig2_path}")
    plt.show()
    
    # ==================== 图3: 统计分布（箱线图 + 小提琴图） ====================
    print("\n" + "=" * 60)
    print("正在绘制图3: 统计分布...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 3.1 箱线图
    bp = axes[0, 0].boxplot([df[ind] for ind in indicators], 
                            labels=indicators, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0, 0].set_title('Box Plot: Distribution of Risk Indicators', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=15)
    
    # 3.2 小提琴图
    parts = axes[0, 1].violinplot([df[ind] for ind in indicators], 
                                  positions=range(1, len(indicators)+1),
                                  showmeans=True, showmedians=True)
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    axes[0, 1].set_xticks(range(1, len(indicators)+1))
    axes[0, 1].set_xticklabels(indicators, rotation=15)
    axes[0, 1].set_title('Violin Plot: Distribution Density', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3.3 直方图（叠加）
    for indicator, color in zip(indicators, colors):
        axes[1, 0].hist(df[indicator], bins=50, alpha=0.5, 
                        label=indicator, color=color, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_title('Histogram: Frequency Distribution', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend(loc='best', fontsize=9)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 3.4 热力图（相关性）
    corr_matrix = df[indicators].corr()
    im = axes[1, 1].imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(indicators)))
    axes[1, 1].set_yticks(range(len(indicators)))
    axes[1, 1].set_xticklabels(indicators, rotation=45, ha='right')
    axes[1, 1].set_yticklabels(indicators)
    axes[1, 1].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(indicators)):
        for j in range(len(indicators)):
            text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)
    
    fig.suptitle('Statistical Distribution Analysis', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, 'fig3_statistical_distribution.png')
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图3已保存: {fig3_path}")
    plt.show()
    
    # ==================== 完成 ====================
    print("\n" + "=" * 60)
    print("所有图表绘制完成！")
    print(f"保存位置: {output_dir}")
    print("文件列表:")
    print(f"  1. fig1_individual_timeline.png")
    print(f"  2. fig2_overlay_comparison.png")
    print(f"  3. fig3_statistical_distribution.png")
    print("=" * 60)
    
    return df

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 运行可视化
    df = load_and_visualize('./indicatorFile/risk_indicator_data.pkl')