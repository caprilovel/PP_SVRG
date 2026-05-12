# %%
# ── 公共导入 & 工具函数 ───────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import os

METRICS = [
    ('train_loss', 'Train Loss',      'Loss'),
    ('eval_loss',  'Eval Loss',       'Loss'),
    ('train_acc',  'Train Accuracy',  'Accuracy'),
    ('eval_acc',   'Eval Accuracy',   'Accuracy'),
    ('weights',    'Weights Norm',    r'$\|\nabla \mathcal{J}(\widetilde{\boldsymbol{\theta}}^s; \widetilde{\boldsymbol{\theta}}^s)\|^2$'),
    ('grads',      'Gradients Norm',  r'$\|\nabla \mathcal{J}(\widetilde{\boldsymbol{\theta}}^s; \widetilde{\boldsymbol{\theta}}^s)\|^2$'),
]

# NeurIPS-style color palette: blue, red first (matching plot_single_result convention)
NEURIPS_COLORS = [
    '#1F77B4',  # blue
    '#D62728',  # red
    '#2CA02C',  # green
    '#FF7F0E',  # orange
    '#9467BD',  # purple
    '#8C564B',  # brown
    '#E377C2',  # pink
    '#17BECF',  # cyan
]


def plot_runs(output_dirs, pdf_dir, custom_legends=None, max_epochs=None):
    """
    读取多个 run 的 train_stats.csv，对每个 metric 各保存一张 PDF。

    Args:
        output_dirs:    run 目录列表
        pdf_dir:        PDF 输出目录
        custom_legends: 与 output_dirs 一一对应的 legend 列表，None 则自动解析
        max_epochs:     只显示 epoch <= max_epochs 的数据，None 表示全部
    """
    def parse_label(path):
        parts = os.path.basename(path).split('_')
        return '_'.join(parts[1:6])

    runs = []
    for d in output_dirs:
        csv_path = os.path.join(d, 'train_stats.csv')
        df = pd.read_csv(csv_path, index_col=0)
        if max_epochs is not None:
            df = df[df['epoch'] <= max_epochs]
        runs.append({'label': parse_label(d), 'df': df})
        print(f"Loaded: {parse_label(d)}  ({len(df)} epochs)")

    os.makedirs(pdf_dir, exist_ok=True)
    for metric_col, metric_title, y_label in METRICS:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, run in enumerate(runs):
            df = run['df']
            if metric_col in df.columns:
                label = (custom_legends[i]
                         if custom_legends and i < len(custom_legends)
                         else run['label'])
                color = NEURIPS_COLORS[i % len(NEURIPS_COLORS)]
                ax.plot(df['epoch'], df[metric_col], label=label,
                        color=color, linewidth=2.5)
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6)
        ax.tick_params(axis='both', which='minor', width=1.5, length=4)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        loc = 'lower right' if metric_col in ('train_acc', 'eval_acc') else 'upper right'
        ax.legend(fontsize=12, loc=loc)
        plt.tight_layout()
        pdf_path = os.path.join(pdf_dir, f'{metric_col}.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"Saved: {pdf_path}")
        plt.close()


def plot_sgd_vs_svrg(svrg_path, sgd_path, pdf_dir, max_epochs=None):
    """
    对同一 temperature 的 SVRG 和 SGD 做对比，保存各 metric 的 PDF。
    max_epochs: 若指定，只显示 epoch <= max_epochs 的数据。
    """
    plot_runs(
        output_dirs=[svrg_path, sgd_path],
        pdf_dir=pdf_dir,
        custom_legends=['SPRINT', 'SGD'],
        max_epochs=max_epochs,
    )


# %%
# Q1 MNIST — different batchsize
plot_runs(
    output_dirs=[
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-085525_SVRG_MNIST_one_layer_Temperature50_lr0.003_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-193800_SVRG_MNIST_one_layer_Temperature50_lr0.003_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-095350_SVRG_MNIST_one_layer_Temperature50_lr0.003_seed2025",
    ],
    pdf_dir='./q1/mnist/',
    custom_legends=[
        'SPRINT (batchsize=64)',
        'SPRINT (batchsize=128)',
        'SPRINT (batchsize=256)',
    ],
    max_epochs=None,
)


# %%
# Q2 MNIST — large alpha (temperature)
plot_runs(
    output_dirs=[
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260326-215528_SVRG_MNIST_one_layer_Temperature80.0_lr0.003_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-084150_SVRG_MNIST_one_layer_Temperature150_lr0.003_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260326-233620_SVRG_MNIST_one_layer_Temperature200_lr0.003_seed2025",
    ],
    pdf_dir='./q2/mnist/',
    custom_legends=[
        'SPRINT (alpha=80)',
        'SPRINT (alpha=150)',
        'SPRINT (alpha=200)',
    ],
    max_epochs=None,
)


# %%
# Q3 MNIST — partial gradient (n_samples)
plot_runs(
    output_dirs=[
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-085525_SVRG_MNIST_one_layer_Temperature50_lr0.003_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260326-220714_SVRG_MNIST_one_layer_Temperature50.0_lr0.003_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-193612_SGD_MNIST_one_layer_Temperature50_lr0.003_seed2025",
    ],
    pdf_dir='./q3/mnist/',
    custom_legends=[
        'SPRINT (full gradient)',
        'SPRINT (n_samples=256)',
        'SPRINT (n_samples=512)',
    ],
    max_epochs=None,
)


# %%
# Q4 MNIST — SGD vs SPRINT per temperature
plot_sgd_vs_svrg(
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-215243_SVRG_MNIST_one_layer_Temperature20_lr0.003_seed2025",
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-225506_SGD_MNIST_one_layer_Temperature20_lr0.003_seed2025",
    pdf_dir='./q4/mnist/Temperature20/',
    max_epochs=None,
)

plot_sgd_vs_svrg(
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-215247_SVRG_MNIST_one_layer_Temperature50_lr0.003_seed2025",
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-225450_SGD_MNIST_one_layer_Temperature50_lr0.003_seed2025",
    pdf_dir='./q4/mnist/Temperature50/',
    max_epochs=None,
)

plot_sgd_vs_svrg(
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-215303_SVRG_MNIST_one_layer_Temperature80_lr0.003_seed2025",
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-225431_SGD_MNIST_one_layer_Temperature80_lr0.003_seed2025",
    pdf_dir='./q4/mnist/Temperature80/',
    max_epochs=None,
)


# %%
# Q1 credit — different batchsize
plot_runs(
    output_dirs=[
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260326-132848_SVRG_credit_mlp_Temperature0.5_lr0.001_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260326-132855_SVRG_credit_mlp_Temperature0.5_lr0.001_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260326-132859_SVRG_credit_mlp_Temperature0.5_lr0.001_seed2025",
    ],
    pdf_dir='./q1/credit/',
    custom_legends=[
        'SPRINT (batchsize=1)',
        'SPRINT (batchsize=32)',
        'SPRINT (batchsize=128)',
    ],
    max_epochs=70,
)


# %%
# Q2 credit — large alpha (temperature)
plot_runs(
    output_dirs=[
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260326-132900_SVRG_credit_mlp_Temperature0.5_lr0.001_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-101837_SVRG_credit_mlp_Temperature1.0_lr0.001_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-194451_SVRG_credit_mlp_Temperature2.0_lr0.001_seed2025",
    ],
    pdf_dir='./q2/credit/',
    custom_legends=[
        'SPRINT (alpha=0.5)',
        'SPRINT (alpha=1.0)',
        'SPRINT (alpha=2.0)',
    ],
    max_epochs=70,
)


# %%
# Q3 credit — partial gradient (n_samples)
plot_runs(
    output_dirs=[
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-171905_SVRG_credit_mlp_Temperature1.0_lr0.001_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-101837_SVRG_credit_mlp_Temperature1.0_lr0.001_seed2025",
        "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-102302_SVRG_credit_mlp_Temperature1.0_lr0.001_seed2025",
    ],
    pdf_dir='./q3/credit/',
    custom_legends=[
        'SPRINT (full gradient)',
        'SPRINT (n_samples=256)',
        'SPRINT (n_samples=512)',
    ],
    max_epochs=70,
)


# %%
# Q4 credit — SGD vs SPRINT per temperature
plot_sgd_vs_svrg(
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-214350_SVRG_credit_mlp_Temperature0.01_lr0.003_seed2025",
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-225643_SGD_credit_mlp_Temperature0.01_lr0.003_seed2025",
    pdf_dir='./q4/credit/Temperature0.01/',
    max_epochs=70,
)

plot_sgd_vs_svrg(
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-214413_SVRG_credit_mlp_Temperature0.2_lr0.003_seed2025",
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-225715_SGD_credit_mlp_Temperature0.2_lr0.003_seed2025",
    pdf_dir='./q4/credit/Temperature0.2/',
    max_epochs=70,
)

plot_sgd_vs_svrg(
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-214814_SVRG_credit_mlp_Temperature0.4_lr0.003_seed2025",
    "/home/zhu.3723/code/code/PP_SVRG/outputs/20260327-225808_SGD_credit_mlp_Temperature0.4_lr0.003_seed2025",
    pdf_dir='./q4/credit/Temperature0.4/',
    max_epochs=70,
)


# # %%
# python credit.py device=cuda:2 dataset=credit nn_model=mlp ratio=1 log=true optimizer=SGD temperature=0.01 batch_size=5 lr=0.003

# # %%
# python main.py device=cuda:2 optimizer=SVRG dataset=MNIST nn_model=one_layer lr=0.003 ratio=0.1 temperature=20 batch_size=100 log=true
