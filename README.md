## k均值聚类 k-mean算法
### 伪码
创建k个点作为初始的质心点（随机选择）  
当任意一个点的簇分配结果发生改变时  
    &nbsp;&nbsp;&nbsp;&nbsp;对数据集中的每一个数据点  
     &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   对每一个质心  
      &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;      计算质心与数据点的距离  
       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 将数据点分配到距离最近的簇  
    &nbsp;&nbsp;&nbsp;&nbsp;对每一个簇，计算簇中所有点的均值，并将均值作为质心  
