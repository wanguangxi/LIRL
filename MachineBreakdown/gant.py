import random
import matplotlib.pyplot as plt

plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["font.sans-serif"] = ["Times new Roman"]
plt.rcParams["font.size"] = 10
plt.rcParams["text.color"] = "black"
plt.rcParams["axes.unicode_minus"] = False

COLOUR_BITS = ['1', '2', '3', '4', '5', '6',
               '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']


def colourGen(workpiece: list) -> list:
    ''' 为每工件对应生成颜色
        @param     workpiece    工件号
        @return    colours      各工件对应的颜色列表
    '''
    colours = []
    workpieceNum = len(set(workpiece))
    for i in range(workpieceNum):
        colourBits = ['#']
        colourBits.extend(random.sample(COLOUR_BITS, 6))
        colours.append(''.join(colourBits))
    return colours

def stable_sort(lists):
    # 添加辅助排序键
    def add_position(item):
        # 添加原始位置作为辅助键
        return item[0], item[1], item[2], item[3]

    # 对辅助键进行排序
    sorted_lists = sorted(lists, key=add_position)

    return sorted_lists
def gantt(macInfo: list, flow: list, macStartTime: list, workpiece: list, operation: list):
    ''' @param    macInfo         对应的机器号 M1=1, M1=2, ...
        @param    flow            各工序加工时间
        @param    macStartTime    各工序开始时间
        @param    workpiece       工件号 J1=1, J2=2, ...
        @param    operation       操作序号
    '''
    colours = colourGen(workpiece)

    
    # 将多个列表合并为一个列表
    combined_list = list(zip(macInfo, flow, macStartTime, workpiece,operation))

    # 使用稳定排序函数进行排序
    sorted_list = stable_sort(combined_list)

    # 分离排序后的列表
    macInfo, flow, macStartTime, workpiece, operation= zip(*sorted_list)
    zipped = zip(flow, macStartTime)
    result =  max(x + y for x, y in zipped)
     

    for i, v in enumerate(macInfo):
        m = v - 1
        plt.barh(m, flow[i], 0.3, left=macStartTime[i],
                 color=colours[workpiece[i]-1])
        plt.text(macStartTime[i] + flow[i] / 8, m, 'J%s.%s' %
                 (workpiece[i], operation[i]), size=8)
    M = len(set(macInfo))    
    label_list = ["Machine" + str(i) for i in range(1, M+1)]
    plt.axvline(x=result, c='k', ls='--', lw=0.5)  # 参考线
    plt.text(result, 0, 'x='+ f'{result:.2f}', ha='left', va='bottom')
    plt.yticks(range(M), label_list)


if __name__ == "__main__":
    MS = [3, 3, 3, 2, 2, 2, 1, 1, 1]
    T = [3, 5, 3, 3, 3, 3, 3, 2, 4]
    macStartTime = [3, 8, 13, 0, 5, 8, 0, 3, 5]
    J = [2, 1, 3, 3, 1, 2, 2, 1, 3]
    oper = [2, 3, 3, 1, 2, 3, 1, 1, 2]

    # 画图
    gantt(MS, T, macStartTime, J, oper)

    plt.show()
