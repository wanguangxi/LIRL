import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
 

# 设置全局字体属性

plt.rcParams['font.size'] = 20          # 字体大小

plt.rcParams['font.family'] = 'Arial'   # 字体类型

plt.rcParams['axes.titlesize'] = 18     # 轴标题大小

plt.rcParams['axes.labelsize'] = 18     # 轴标签大小

plt.rcParams['legend.fontsize'] = 18    # 图例字体大小
fig = plt.figure(1)


def get_data():
    '''获取数据
    '''
    try:
        basecond = np.load(r'Algorithm/breakdown/job10_score_records_0.5_0.5_co_breakdown.npy')
        cond1 = np.load(r"Algorithm/breakdown/job10_score_cae_records_0.5_0.5_co_breakdown.npy")
        cond2 = np.load(r"Algorithm/breakdown/job10_score_cae_records_0.5_0.5_co_breakdown+.npy")
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
        print("Please check that the required .npy files exist at the specified paths.")
        raise
    return basecond, cond1, cond2

color_palette = {

    'LIRL-our': '#EC5b5b',  # 红色调

    'HyAR': '#3ABBCE',  # 绿色调

    'HyAR+': '#FFC839'   # 蓝色调
}
data = get_data()
label = ['LIRL-our', 'HyAR', 'HyAR+']
df=[]
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='reward'))
    df[i]["Algorithm"]= label[i]


plt.subplot(2, 3, 1)    
df=pd.concat(df) # 合并
sns.lineplot(x="episode", y="reward", hue="Algorithm", palette=color_palette,data=df)
plt.title(r'Robot breakdown:T(10)$\times$R(5), $\alpha=0.5,\beta=0.5$',  fontsize=20, fontweight='bold')
plt.xlabel('Episode', fontsize=18, fontweight='bold')
plt.ylabel('Performance', fontsize=18, fontweight='bold')
# plt.legend(title='Category', title_fontsize='14', labelspacing=0.5)
plt.legend(loc=4)


def get_data():
    '''获取数据
    '''
    basecond = np.load(r"Algorithm/breakdown/job20_score_records_0.5_0.5_co_breakdown.npy")
    cond1 = np.load(r"Algorithm/breakdown/job20_score_cae_records_0.5_0.5_co_breakdown.npy")
    cond2 = np.load(r"Algorithm/breakdown/job20_score_cae_records_0.5_0.5_co_breakdown+.npy")
    return basecond, cond1, cond2


data = get_data()
label = ['LIRL-our', 'HyAR', 'HyAR+']
df=[]
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='reward'))
    df[i]["Algorithm"]= label[i]


plt.subplot(2, 3, 2)    
df=pd.concat(df) # 合并
sns.lineplot(x="episode", y="reward", hue="Algorithm", palette=color_palette,data=df)
plt.title(r'Robot breakdown:T(20)$\times$R(5), $\alpha=0.5,\beta=0.5$',  fontsize=20, fontweight='bold')
plt.xlabel('Episode', fontsize=18, fontweight='bold')
plt.ylabel('Performance', fontsize=18, fontweight='bold')
# plt.legend(title='Category', title_fontsize='14', labelspacing=0.5)
plt.legend(loc=4)



def get_data():
    '''获取数据
    '''
    basecond = np.load(r"Algorithm/breakdown/job50_score_records_0.5_0.5_co_breakdown.npy")
    cond1 = np.load(r"Algorithm/breakdown/job50_score_cae_records_0.5_0.5_co_breakdown.npy")
    cond2 = np.load(r"Algorithm/breakdown/job50_score_cae_records_0.5_0.5_co_breakdown+.npy")
    return basecond, cond1, cond2


data = get_data()
label = ['LIRL-our', 'HyAR', 'HyAR+']
df=[]
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='reward'))
    df[i]['Algorithm']= label[i]


plt.subplot(2, 3, 3)   

df=pd.concat(df) # 合并
sns.lineplot(x="episode", y="reward", hue="Algorithm", palette=color_palette,data=df)
plt.title(r'Robot breakdown:T(50)$\times$R(5), $\alpha=0.5,\beta=0.5$',  fontsize=20, fontweight='bold')
plt.xlabel('Episode', fontsize=18, fontweight='bold')
plt.ylabel('Performance', fontsize=18, fontweight='bold')
# plt.legend(title='Category', title_fontsize='14', labelspacing=0.5)
plt.legend(loc=4)



def get_data():
    '''获取数据
    '''
    basecond = np.load(r"Algorithm/processtime/job10_score_records_0.5_0.5_co.npy")
    cond1 = np.load(r"Algorithm/processtime/job10_score_cae_records_0.5_0.5_co.npy")
    cond2 = np.load(r"Algorithm/processtime/job10_score_cae_records_0.5_0.5_co+.npy")
    return basecond, cond1, cond2

data = get_data()
label = ['LIRL-our', 'HyAR', 'HyAR+']
df=[]
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='reward'))
    df[i]["Algorithm"]= label[i]


plt.subplot(2, 3,4)    
df=pd.concat(df) # 合并
sns.lineplot(x="episode", y="reward", hue="Algorithm", palette=color_palette,data=df)
plt.title(r'Uncertain duration :T(10)$\times$R(5), $\alpha=0.5,\beta=0.5$',  fontsize=20, fontweight='bold')
plt.xlabel('Episode', fontsize=18, fontweight='bold')
plt.ylabel('Performance', fontsize=18, fontweight='bold')
# plt.legend(title='Category', title_fontsize='14', labelspacing=0.5)
plt.legend(loc=4)


def get_data():
    '''获取数据
    '''
    basecond = np.load(r"Algorithm/processtime/job20_score_records_0.5_0.5_co.npy")
    cond1 = np.load(r"Algorithm/processtime/job20_score_cae_records_0.5_0.5_co.npy")
    cond2 = np.load(r"Algorithm/processtime/job20_score_cae_records_0.5_0.5_co+.npy")
    return basecond, cond1, cond2

data = get_data()
label = ['LIRL-our', 'HyAR', 'HyAR+']
df=[]
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='reward'))
    df[i]["Algorithm"]= label[i]


plt.subplot(2, 3, 5)    
df=pd.concat(df) # 合并
sns.lineplot(x="episode", y="reward", hue="Algorithm", palette=color_palette,data=df)
plt.title(r'Uncertain duration :T(20)$\times$R(5), $\alpha=0.5,\beta=0.5$', fontsize=20, fontweight='bold')
plt.xlabel('Episode', fontsize=18, fontweight='bold')
plt.ylabel('Performance', fontsize=18, fontweight='bold')
# plt.legend(title='Category', title_fontsize='14', labelspacing=0.5)
plt.legend(loc=4)


def get_data():
    '''获取数据
    '''
    basecond = np.load(r"Algorithm/processtime/job50_score_records_0.5_0.5_co.npy")
    cond1 = np.load(r"Algorithm/processtime/job50_score_cae_records_0.5_0.5_co.npy")
    cond2 = np.load(r"Algorithm/processtime/job50_score_cae_records_0.5_0.5_co+.npy")
    return basecond, cond1, cond2

data = get_data()
label = ['LIRL-our', 'HyAR', 'HyAR+']
df=[]
for i in range(len(data)):
    df.append(pd.DataFrame(data[i]).melt(var_name='episode',value_name='reward'))
    df[i]["Algorithm"]= label[i]


plt.subplot(2, 3, 6)    
df=pd.concat(df) # 合并
sns.lineplot(x="episode", y="reward", hue="Algorithm", palette=color_palette,data=df)
plt.title(r'Uncertain duration :T(50)$\times$R(5), $\alpha=0.5,\beta=0.5$', fontsize=20, fontweight='bold')
plt.xlabel('Episode', fontsize=18, fontweight='bold')
plt.ylabel('Performance', fontsize=18, fontweight='bold')
# plt.legend(title='Category', title_fontsize='14', labelspacing=0.5)
plt.legend(loc=4)

plt.figure(figsize=(8, 6))
# ...绘图代码...

# 自动调整边界和子图间距
plt.tight_layout(
    pad=0.8,      # 整体边距 (默认1.08)
    w_pad=0.5,     # 子图水平间距 (默认0.5)
    h_pad=1.0,     # 子图垂直间距 (默认1.0)
    rect=(0.2, 0.2, 0.2, 0.2)  # 图形在画布中的位置矩形 [left, bottom, right, top]
)
fig.savefig('output.svg', format='svg')


print("图形已保存为 SVG 文件：output.svg")
plt.show()


