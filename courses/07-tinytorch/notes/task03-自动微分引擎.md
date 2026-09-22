# Task 3：自动微分引擎（Autograd）

> 对应 Issue #22：从 Tensor 到 Transformer——构建 GPT 模型 | Task 3 自动微分引擎（截止 09-23 中午 12:00）
> 学习资料：mlsysbook.ai/tinytorch/modules/06_autograd.html
> 实跑环境：本地 Windows · PowerShell 5.1 · Python 3.10.2 · NumPy 2.2.6 · pytest 9.1.1（纯 NumPy 实现，无需 GPU）

## 0. 任务与产出对照

| Issue 要求 | 落点 |
|---|---|
| 实现 Function 基类（forward/backward），为每个张量操作注册反向传播函数 | `src/06_autograd/06_autograd.py` 的 `Function` 基类 + 20 个 `*Backward` 子类 + `enable_autograd()` 的算子打补丁表 |
| 实现计算图的构建和链式法则的遍历 | `Tensor._grad_fn` 建图 + `Tensor.backward()` 的递归反向遍历（含广播归约、梯度累加、图释放） |
| 4.1 实现思路 + 测试截图 | 见第 3、6 节；截图 `evidence/task3/task3_06_autograd_runshot.png` 等 3 张 |
| 4.2 requires_grad / grad_fn / backward() 协作 | 见第 4 节 |
| 4.3 为什么 PyTorch 计算图是动态的 | 见第 5 节 |
| 4.4 计算图可视化示意图 | 见第 7 节（ASCII 图 + 实跑验证） |

一句话定位：Module 01–05 只会"算前向"，Module 06 给每个运算装上反向规则，让 `loss.backward()` 能自动把所有参数的梯度算出来——这是"训练"从手动推导变成一行调用的分水岭。

## 1. 实现清单：三层结构

整个 autograd 引擎分三层，职责清晰：

**① Function 基类（梯度规则引擎）**

```python
class Function:
    def __init__(self, *tensors):
        self.saved_tensors = tensors     # 保存反向需要的输入/中间量
    def apply(self, grad_output):
        raise NotImplementedError        # 子类实现具体梯度规则
```

每个可微运算一个子类，实现链条式法则的"局部导数"：

| 算子（前向） | Backward 子类 | 反向规则 |
|---|---|---|
| `a + b` | AddBackward | grad_a = grad_z, grad_b = grad_z |
| `a * b` | MulBackward | grad_a = grad_z·b, grad_b = grad_z·a（乘积法则） |
| `a - b` / `a / b` | SubBackward / DivBackward | ±1 透传；`1/b`、`-a/b²` |
| `A @ B` | MatmulBackward | grad_A = grad_Z·Bᵀ，grad_B = Aᵀ·grad_Z（3D+ 只交换最后两维，保留 batch 维） |
| `x.T` / `permute` / `reshape` / `getitem` | Transpose/Permute/Reshape/SliceBackward | 逆重排 + 零填充还原形状 |
| `sum(axis)` | SumBackward | 沿归约轴广播回原形状 |
| ReLU/Sigmoid/Tanh/Softmax/GELU | *Backward | 门控（负半轴 0）；σ(1-σ)；1-tanh²；softmax 的雅可比形式 `grad - y·Σgrad` |
| MSE / BCE / CrossEntropy | *Backward | `2(pred-target)/N`；`-t/p + (1-t)/(1-p)`；`softmax - one_hot` |

**② 计算图构建（运行时打补丁，而不是包一层新类）**

`enable_autograd()`  monkey-patch 两处：

- Tensor 的 `__add__/__radd__/__sub__/__mul__/__matmul__/__truediv__/__getitem__/sum/transpose/reshape`：
  任一输入 `requires_grad=True` 时，结果张量自动 `requires_grad=True` 且 `_grad_fn = XxxBackward(输入们)`；
- Module 02/04 的激活与损失 `forward`（ReLU/Sigmoid/Tanh/Softmax/GELU、MSE/BCE/CE）：
  因为激活类内部是 `np.maximum(0, x.data)` 这类裸 NumPy，会"截断"计算图，必须单独挂 `_grad_fn`。
  课程里有一段回归测试专门盯这件事：Tanh 曾因漏打补丁导致 `tanh(x).backward()` 静默不产出梯度。

**③ 反向遍历（链式法则执行器）**

```python
def backward(self, gradient=None, retain_graph=False):
    if gradient is None:                       # 标量损失：种子 dL/dL = 1
        gradient = np.ones_like(self.data)
    self.grad += gradient                      # 梯度累加（多路径自然求和的前提）
    grad_fn = self._grad_fn
    if grad_fn is not None:
        grads = grad_fn.apply(gradient)        # 局部导数 × 上游梯度
        for tensor, grad in zip(grad_fn.saved_tensors, grads):
            if tensor.requires_grad and grad is not None:
                tensor.backward(grad, retain_graph)   # 沿图递归
    if not retain_graph:
        self._grad_fn = None                   # 释放计算图，防内存随步数线性增长
```

配套零件：`_reduce_broadcast_grad(grad, shape)`（广播逆运算：先对多出的前导维 `sum(axis=0)`，再对原尺寸为 1 的维 `keepdims` 求和）、`zero_grad()`、`is_grad_enabled()` 与 `no_grad` 上下文。

## 2. 实现中的关键设计（为什么这样写）

1. **反向模式（reverse-mode）而非数值/符号微分**：前向一遍 + 反向一遍即得全部参数梯度，代价 O(输入数)×常数；数值差分要 N+1 遍前向且精度差；符号微分表达式爆炸。
2. **每个 Function 自带 saved_tensors**：乘积法则需要"另一个输入"，matmul 需要转置对方，Sigmoid/Tanh 需要前向输出——反向要什么就在前向存什么，这是 autograd 的主要内存开销来源（反向完立刻释放）。
3. **梯度累加 `+=` 而非赋值**：同一参数被多条路径（或多次 backward）共享时，链式法则要求求和；这也决定了"每个迭代开始前要 zero_grad"。
4. **广播必须可逆**：前向 `(32,128)+(128,)` 背地里把 bias 复制了 32 份，反向就必须把这 32 份的梯度求和回 `(128,)`——这是 autograd 实现里最容易错的一环。
5. **动态图 = 每步重建**：`_grad_fn` 即图边，backward 后置 None 即释放；内存与执行同步，见第 5 节。

## 3. 4.1 Autograd 实现思路（打卡版）

前向计算时，每个可微运算除计算结果外，还把「输入张量 + 局部求导规则 + 反向所需中间量」写进输出 Tensor 的 `_grad_fn`（指向对应 `Function` 实例），于是一条 DAG 在执行过程中被即时构建。反向传播从标量损失开始，以 `∂L/∂L = 1` 为种子，按 `_grad_fn` 链逐层回传，每一步执行 `上游梯度 × 局部导数`，落到叶子节点时 `grad +=` 累加。广播维度还原、梯度累加、中间量保存与建图/释放是四个必须处理的工程点。

## 4. 4.2 requires_grad / grad_fn / backward() 的协作关系

三者分工可以概括为「开关、账本、执行器」：

- **requires_grad（开关）**：标记这个 Tensor 是否参与梯度跟踪。叶子参数（W、b）手动置 True；运算结果由 `enable_autograd` 按输入自动推导。全局开关是 `is_grad_enabled()`（`no_grad` 上下文可临时关闭建图，推理时省内存）。
- **grad_fn / _grad_fn（账本）**：非叶子 Tensor 的属性，指向创建它的 `Function` 对象，内含 `saved_tensors`（前向输入/中间值）与梯度规则。叶子 Tensor 的 `_grad_fn` 恒为 None——这是区分"参数"和"中间激活"的标志。
- **backward()（执行器）**：入口。标量调用时种子为 1，沿 `_grad_fn` 递归回传，梯度累加到叶子 `.grad`；默认反向后释放图（`retain_graph=True` 可保留）。

实跑验证（`demo_autograd.py` 第 [1] 节，见截图 `task3_demo_runshot.png`）：

```text
leaf  x : requires_grad=True  _grad_fn=None            # 参数
mid    y : requires_grad=True  _grad_fn=MulBackward    # x*3
out    z : requires_grad=True  _grad_fn=AddBackward    # y+1
before backward: x.grad = None
after  z.backward(): x.grad = [3.]   （链式法则 dL/dx = 1×3 = 3）
graph released after backward: y._grad_fn = None
```

因为 `grad +=` 是累加语义，PyTorch/TinyTorch 训练循环的标准写法都是 `optimizer.zero_grad() → loss.backward() → optimizer.step()`，不清零就会把上一个 iteration 的梯度带进来。

## 5. 4.3 为什么 PyTorch 的计算图是动态的？对框架设计的影响

**为什么动态**：PyTorch 采用 define-by-run——图不是先声明再执行，而是在前向代码**真正跑到哪一行**时临时建一个节点。

```python
for i in range(n):                 # n 可以取决于数据
    if cond(x):  y = f1(x)         # 图结构 A
    else:        y = f2(x)         # 图结构 B
    loss = y.mean(); loss.backward()
```

数学上可写成 `G_t = Trace(Forward(x_t))`：每次前向都在重新"录制"一张新图，因此 if/for/while、可变长度序列、数据依赖的控制流都能自然参与计算，且每一步的中间值可以用普通 Python 断点/print 检查（静态图要靠 tf.Session 类专用工具）。

**对框架设计的影响**：

| 维度 | 动态图的要求 |
|---|---|
| 内存管理 | 不能编译期统一规划；必须在运行时跟踪每个中间张量生命周期——这就是 `saved_tensors` + backward 后释放图的原因，大模型还要靠 activation checkpointing 缓解 OOM |
| 性能 | 每步重建图节点有运行时可微性与 Python 开销；靠高度优化的 C++/CUDA 后端、内存池、预分配，以及 `torch.compile`（把运行路径 trace 下来再优化）弥补 |
| 编译器 | 优化动态图必须先捕获运行路径、建立输入 shape 的守卫（guard），遇到无法捕获的 graph break 只能退回 eager |
| 分布式 | 图结构每次迭代都可能不同，通信/调度无法提前静态规划 |
| 调试体验 | 图即代码执行，打印即所见——这是 PyTorch 取代 TF1.x 成为研究主流的关键原因 |

本模块的 `backward()` 就体现了动态图的工程取舍：图是"边跑边建、用完即焚"的，内存峰值 = 一条路径上的激活量；`retain_graph`、`no_grad` 这些开关本质上都是在"保留多少运行时状态"上做权衡。

## 6. 测试结果（全部实跑，非教程样例）

| 套件 | 结果 |
|---|---|
| 模块自测 `test_module()`（`python run_selftest.py 06`） | 🎉 ALL TESTS PASSED! Module ready for export. |
| `test_06_autograd_progressive.py` | 17 passed |
| `test_autograd_core.py` | 18 skipped（文件头注明：Legacy Variable API placeholder，覆盖已迁到现行/集成测试） |
| `test_autograd_gradient_flow.py` | 9 passed |
| `test_batched_matmul_backward.py` | 3 passed |
| `test_gradient_correctness.py` | 24 passed |
| **Module 06 合计** | **53 passed, 18 skipped** |
| **01–06 全量回归**（确认打补丁没有破坏既有模块） | **239 passed, 18 skipped** |

另用自写证据脚本 `demo_autograd.py` 做了 7 组数学/语义校验（截图 `task3_demo_runshot.png`）：

1. **解析梯度 vs 数值梯度**：`L = Σ tanh(x@W)·v`，float64 中心差分参考实现，5 维梯度逐项对照 `max abs err = 9.32e-08`；
2. **广播归约**：`x(4,3) + b(3,)`，上游梯度非均匀时 `b.grad = 每列求和 [18,22,26]`，`x.grad` 原样透传；
3. **菱形多路径累加**：`Σ((x·2)+(x·3))` → `dL/dx = 2+3 = 5`；
4. **zero_grad / retain_graph 语义**：累加行为、图保留后两次 backward = 7+7=14；
5. **手算对照**：`z=x@W; y=z+b; L=Σ(y·c)` → `db=c`、`dW=xᵀcᵀ`、`dx=cᵀWᵀ` 三条全部一致；
6. **训练闭环**：2 层网络拟合 XOR，手动 SGD 200 步，MSE 0.4608 → **0.00117**——引擎可直接用于训练。

### 实测发现的两个真实局限（如实记录）

1. **同一中间节点重复入图时梯度丢失**：`L = (y*y).sum()` 或最小复现 `b = a + a`（a 为中间节点）时，`saved_tensors` 里同一对象出现两次；第一次递归回传后 `_grad_fn=None` 释放了图，第二条路径无法继续回传，实测梯度恰为正确值的 1/2。PyTorch 用**拓扑排序 + 每节点梯度缓冲 + 消费者引用计数**解决（所有消费者的梯度都到齐后才释放）。
2. **`Tensor.mean()` 不在跟踪列表**：`enable_autograd` 只 patch 了 `sum`，`mean()` 内部 `np.mean(self.data)` 返回裸 Tensor，会让图在此处断开（`diff*diff).mean()` 反传后所有参数 `.grad` 为 None）。规避：用 `sum()` 后除 N，或用 Module 04 的 `MSELoss`（其 forward 被单独打了补丁挂 `MSEBackward`）。

这两个点不在课程测试覆盖范围内（53 个测试全过），但属于"实现计算图遍历"这个任务的真实边界，写在这里备查。

## 7. 4.4 计算图可视化示意图

以前向 `z = x@W; y = z + b; L = Σ(y·c)` 为例（实跑输出见 `task3_demo_runshot.png` 第 [6] 节）：

```text
前向（建图，实线）
  x(1,2) ──┐
           ├─[matmul]──► z(1,3) ──┐
  W(2,3) ──┘                     ├─[+b]──► y(1,3) ──[·c, Σ]──► L(标量)
                       b(3,) ─────┘            c(3,) 常量(requires_grad=False)

_grad_fn 链: L(SumBackward) ← y(AddBackward) ← z(MatmulBackward)

反向（梯度流，虚线；种子 dL/dL = 1）
  ∇x = [[2.5, 2.0]]   ∇W = [[1,2,3],[2,4,6]]   ∇b = [1,2,3]
  （与手算 db=c, dW=xᵀcᵀ, dx=cᵀWᵀ 完全一致）
```

叶子/非叶子与生命周期：

```text
节点 x,W,b  : requires_grad=True, _grad_fn=None  → 叶子，梯度落 .grad
节点 z,y    : requires_grad=True, _grad_fn=MatmulBackward/AddBackward → 中间
节点 L      : 标量，backward() 的种子为 1
backward 后 : 每个被访问节点的 _grad_fn 置 None，图释放（retain_graph=True 除外）
```

对应到 PyTorch：`x,W,b` 是叶子（`requires_grad=True, grad_fn=None`），`z,y` 是非叶子（`grad_fn=<MatmulBackward0>/<AddBackward0>`），`L` 上调用 `backward()`。

## 8. 官方 CLI 完成流（tito module complete 06）

官网最新版 Module 06 页面给出的验收标准是 `tito module complete 06` 四步全绿：**单元测试 → 导出到 `tinytorch.core.autograd` → 集成测试 → 进度记录**。本地工作副本是 `src/*.py`（jupytext percent）变体、`modules/` 下没有 notebook，官方 CLI 的导出步骤会报 "Notebook not found"，因此先用 jupytext 把 `src/<NN>/<NN>.py` 转成 `modules/<NN>/<name>.ipynb`，补齐 CLI 依赖（rich / certifi / pyyaml / requests / jupytext / nbdev）后按官网流程跑完（幂等，可重复执行）：

```text
PS> tito module complete 06
   ✅ Unit tests: 13/13 passed                       # 本地变体 notebook 内 13 项单元测试
   📦 Exporting autograd.ipynb → tinytorch/core/autograd.py
   ✅ Integration tests: 17/17 passed                # 与官网 "Integration tests after export (17)" 数量一致
   📈 Progress updated: 6 modules completed
   ✅ Module 06 marked complete
   │ You didn't import Autograd. You BUILT it.
```

顺手把前置模块 01–05 也在同一副本用官方 CLI 跑通（单元测试 24/21/13/18/19，集成测试 10/12/13/12/12），累计 **6/20 modules (30%)**，进度记录由 CLI 自己写入。注意：官方 CLI 的 nbdev 导出会重新生成 `tinytorch/core/autograd.py`（头部变为 `AUTOGENERATED! DO NOT EDIT`，文件与早先 mini nbdev 导出等价），导出后模块测试与全量回归仍然全绿。原始日志 `evidence/task3/task3_tito_complete.log`，截图 `task3_tito_complete_runshot.png`，复现脚本 `code/task3/task3_tito.ps1`。

## 9. 学习资料最新版核对（mlsysbook.ai，2026-09-22 抓取存档）

已通读官网 Module 06 最新页面（存档 `_tmp_tt_docs/06_autograd_latest.{html,txt}`），与本地副本对照：

| 项目 | 官网最新版 | 本地工作副本 |
|---|---|---|
| 学习侧文件 | `modules/06_autograd/autograd.ipynb` | `modules/` 下无 notebook，源码在 `src/06_autograd/06_autograd.py` |
| 要求手写的代码 | 3 个函数：`MatMul.backward`、`ReLUFunction.backward`、`MSEFunction.backward`（`# YOUR CODE HERE` 标记） | 同一 Notebook 的另一种变体：约 20 个 `*Backward` 子类全部给出（`### BEGIN SOLUTION`），覆盖面更广 |
| notebook 单元测试 | 10 项，含 **Gradients Through a Reused Tensor** | 13 项调用（命名集合与官网 10 项不完全相同，语义同类） |
| 导出后集成测试 | **17**（`test_06_autograd_progressive.py`） | 17，数量一致 ✅ |
| 验收命令 | `tito module complete 06` | 已验证通过（见第 8 节） |

官网算例实测（本地引擎逐一复现，输出一致）：

- **Figure 1**：`x=3, y=x*x=9, L=y+x=12` → 反向把直接路径 1 和两条乘法路径 3、3 合并，`dx = 7`，本地引擎输出 **7** ✅（正对应官网所列 "Gradients Through a Reused Tensor"：**叶子**被复用的情况）；
- **Quick example**：`y = x*3 + 1` → `x.grad = [3.0]` ✅。

官网 "What you're not building yet" 划定的边界也与实现一致：高阶导数（`create_graph=True`）、动态图控制流捕获、GPU kernel fusion、activation checkpointing 均不在本模块范围；延伸阅读是配套书章节 *Automatic Differentiation: Recording the Graph and Walking It Backward*。

> 结论：本地副本比官网当前版更旧也更全（缺 `modules/*.ipynb`、单元测试命名/条数不同），但官网的验收四步在本地全部跑通，官网两个算例也与本地引擎输出一致；官网第 10 项单元测试 "Gradients Through a Reused Tensor" 针对的叶子复用场景本地通过（dx=7），而**中间节点**重复入图仍会丢第二路梯度（第 6 节局限 1）——这是本地实现相对官网测试集的真实差距，已如实记录。

## 10. 复现方式

```bash
# 一键跑完（环境/导出/自测/pytest/引擎证据 + 截图，日志落 evidence/task3/）
pwsh courses/07-tinytorch/code/task3/task3_verify.ps1

# 官方 CLI 完成流（装依赖 → 生成 notebook → tito complete 01..06 → 截图 → 回归）
pwsh courses/07-tinytorch/code/task3/task3_tito.ps1

# 分步
python courses/07-tinytorch/code/task3/export_autograd.py     # 06 -> tinytorch/core/autograd.py
python courses/07-tinytorch/code/task3/run_selftest.py 06     # 模块自测
python -m pytest tests/06_autograd -q --tb=short              # 53 passed, 18 skipped
python courses/07-tinytorch/code/task3/demo_autograd.py       # 7 组数学/语义校验
python courses/07-tinytorch/code/task3/make_runshot.py <log.txt> <out.png> "<标题>" "<cwd>" "<cmd>" "[meta]"   # 生成打卡截图
```

证据文件（`evidence/task3/`）：`task3_env.txt`、`task3_export.log`、`task3_06_autograd_selftest.txt`、`task3_pytest_06.log`（-v 全量）、`task3_pytest_by_file.log`（逐文件汇总）、`task3_pytest_full_01_06.log`、`task3_autograd_demo.txt`、`task3_tito_complete.log`（官方 CLI 全量输出）、`task3_tito_keylines.log`，以及四张终端风格截图（`*_runshot.png`）。
