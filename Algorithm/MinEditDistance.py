# python实现最小编辑距离
""""""
"""在计算文本的相似性时，经常会用到编辑距离，其指两个字符串之间，由一个字符串转成另一个字符串所需的最少编辑操作次数。
在字符串形式上来说，编辑距离越小，那么两个字符串的相似性越大，暂时不考虑语义上的问题。
其中，编辑操作包括以下三种：
        插入：将一个字符插入某个字符串
        删除：将字符串中的某个字符删除
        替换：将字符串中的某个字符替换为另一个字符
将字符串“batyu”变为“beauty”，编辑距离是多少呢？分析步骤如下：将batyu插入字符e有：beatyu；将beatyu删除字符u有：beaty；
将beaty插入字符u有：beauty，所以，字符串“batyu”与字符串“beauty”之间的编辑距离为：3。"""

"""如何计算两个字符串之间的编辑距离？？？
当两个字符串都为空串的时候，那么编辑距离就是0
当其中一个字符串为空串时，那么编辑距离为另一个非空字符串的长度
当两个字符串A,B均为非空时(假设长度分别为i,j)，那么有如下三种情况，我们取这三种情况的最小值即可：
1).已知字符串A中长为 i - 1(从字符串首开始，以下描述字符串长默认此种描述)和字符串B长为j的编辑距离，那么在此基础上加1即可
2).长度分别为i和j-1的编辑距离已知，那么加1即可
3).长度分别为i-1和j-1的编辑距离已知，此时需要考虑两种情况，若第i个字符和第j个字符不同，那么加1即可，如果相同，就不需要加1.
从上面的描述，很明显可以发现是动态规划的思想。
我们将上面的叙述数学化，则有：求长度为m和n的字符串A、B的编辑距离，即函数：edit(i,j),
它表示第一个长度为i(从字符首开始)的字符串与第二个长度为j的字符串之间的编辑距离。
动态规划表达式则有如下写法，假设i,j分别表示字符串A,B的长度：
if i==0 且 j==0,edit(i,j)=0
if (i==0 且 j>0) 或者 (i>0 且j ==0)，edit(i,j)=i + j
if i>= 1 且 j >= i, edit(i, j) = min(edit(i-1,j) + 1, edit(i, j-1) + 1, edit(i-1,j-1) + d(i,j));
当第一个字符串的第i个字符与第二个字符串第j个字符不相同时，d(i,j)=1，否则为0"""
# 有上述思路后，python代码如下
def min_edit_distance(string1, string2):
    edit = [[i+j for j in range(len(string2)+1)] for i in range(len(string1)+1)]
    for i in range(1, len(string1)+1):
        for j in range(1, len(string2)+1):
            if string1[i-1] == string2[j-1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(edit[i][j-1]+1, edit[i-1][j]+1, edit[i-1][j-1]+d)
    return edit[len(string1)][len(string2)]


distance = min_edit_distance('batyu', 'beauty')
print(distance)