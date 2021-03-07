import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from sklearn.model_selection import learning_curve
#sixth
def plotLearningCurve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    estimator : 分类器
    title : 图的标题
    X : 输入的特征值
    y : 输入的预测值
    ylim : 设定图像中纵坐标的最低点和最高点（元组形式）
    cv : 做交叉验证时数据分成的份数，其中一份作为cv集，其余n-1份作为训练集(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    
    trainScoresMean = np.mean(train_scores, axis=1)
    trainScoresStd = np.std(train_scores, axis=1)
    testScoresMean = np.mean(test_scores, axis=1)
    testScoresStd = np.std(test_scores, axis=1)
    
    if plot:
        mp.rc('font', family='FangSong', weight='bold', size='11')
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("训练样本数")
        plt.ylabel("得分")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.fill_between(train_sizes, trainScoresMean - trainScoresStd, trainScoresMean + trainScoresStd, alpha=0.1, color="b")
        plt.fill_between(train_sizes, testScoresMean - testScoresStd, testScoresMean + testScoresStd, alpha=0.1, color="r")
        plt.plot(train_sizes, trainScoresMean, 'o-', color="b", label="训练集上得分")
        plt.plot(train_sizes, testScoresMean, 'o-', color="r", label="交叉验证集上得分")       
        plt.legend(loc="best")
        plt.show()
    
    midpoint = ((trainScoresMean[-1] + trainScoresStd[-1]) + (testScoresMean[-1] - testScoresStd[-1])) / 2
    diff = (trainScoresMean[-1] + trainScoresStd[-1]) - (testScoresMean[-1] - testScoresStd[-1])
    return midpoint, diff
