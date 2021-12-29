#### Problems

* too large spring stiffness (1e10) --> $cond(A)\gg1$ --> CG doesn't converge
* short term (3 frames)
  * 仿真最后一帧和 target 差太多的时候，会先在第一帧来一个很大的力把布料拉很远，然后再慢慢回去 (应该还是系数的问题)
* large term (50 frame)
  * smooth coeff 太小了导致力太大直接爆炸 (然而系数太大会导致力太小，收敛时也到不了 target position)
    * smooth 系数依赖于控制力的大小，就先有鸡先有蛋了-> 搞一个和系统参数有关的 position loss coeff ($m\over{n_f^2h^4}$) -> demo2 跑通了，但对系数是真滴敏感，0.5的差别就很大
  * Forward 没收敛 ($r=0$ 不成立)
    * 实现了牛顿法，但是迭代次数>1的时候会飞起来 -> 有bug? -> 没有, 是 CG 的跳出 err 设太大了...
    * 所以目前只用的一步牛顿迭代：XPBD 迭代 100 次比 Newton 迭代一次 residual 要大，优化结果newton (~2000) 略好于 XPBD (~3000)
* New optimization algo.
  * **[observation]** new direction is brought by $\lambda$, regularization term is just a scaling of $p$ -> $p^*=p^{(k)}+\alpha {\part L\over{\part q}}{\part q\over{\part p}}, p^{(k+1)}=(1-\epsilon)p^*$
    * 直接 scale p 好像不太行，会导致 loss 的下降比较 random，也许可以再整个 line-search (算了...)
  * Quasi-Newton
  * SAP
* Med2
  * implicit Euler 初始化，symplectic 做优化等式接出来 loss (virtual force) 贼大 -> 所以就用 implicit 啊...
  * forward 初始化
    * 收敛很慢，loss 主要分布在两端，所以一直是在努力降两头的loss -> 只可视化中间的看看？
  * 全0初始化
    * 初始的loss就比 forward simulation 初始化跑200次迭代还小了...
    * 会收敛到 trivial solution (诡异的飘在空中)
    * (LLT 偶尔会 fail)
  * SAP 初始化
    * 两头的 loss 会进一步降低 -> 只可视化中间的看看？-> 中间的也降了，只不过这个 SAP intial guess 对于 loopy loss 来说还是两头的很大
  * soft constrain
    * ？甩的挺高

#### Impl

* SNode 个数有上限，所以不能每帧开一个，要把所有帧的数据放到一个大 field 里
* safe normalization