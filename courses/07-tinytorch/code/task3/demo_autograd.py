# -*- coding: utf-8 -*-
"""Task 3 自动微分引擎证据脚本：验证 autograd 引擎的 API 语义与梯度正确性。

覆盖:
  1. enable_autograd() 之后 requires_grad / _grad_fn / backward() 三者协作
  2. 解析梯度 vs 数值梯度（float64 中心差分）逐项对照
  3. 广播梯度归约 (4,3)+(3,) -> (3,)
  4. 菱形多路径梯度累加
  5. zero_grad / retain_graph 语义
  6. 计算图可视化 + 手算对照；以及同一中间节点重复入图时的已知局限
  7. 微型 2 层网络手写 SGD 训练闭环 (loss 下降)
"""
import numpy as np

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd, is_grad_enabled
from tinytorch.core.activations import ReLU, Tanh
from tinytorch.core.losses import MSELoss

enable_autograd()
print("autograd enabled:", is_grad_enabled())


def head(t):
    print()
    print("=" * 70)
    print(t)
    print("=" * 70)


def gfn(t):
    fn = getattr(t, "_grad_fn", None)
    return type(fn).__name__ if fn is not None else "None"


# ---------------------------------------------------------------- 1
head("[1] requires_grad / _grad_fn / backward() 协作关系")
x = Tensor([2.0], requires_grad=True)
c = 3.0
y = x * c
z = y + 1.0
print("leaf  x : requires_grad=%s _grad_fn=%s" % (x.requires_grad, gfn(x)))
print("mid    y : requires_grad=%s _grad_fn=%s" % (y.requires_grad, gfn(y)))
print("out    z : requires_grad=%s _grad_fn=%s" % (z.requires_grad, gfn(z)))
print("before backward: x.grad =", x.grad)
z.backward()
print("after  z.backward(): x.grad =", x.grad, " (链式法则 dL/dx = 1*3 = 3)")
print("graph released after backward: y._grad_fn =", gfn(y))

# ---------------------------------------------------------------- 2
head("[2] 解析梯度 (autograd) vs 数值梯度 (float64 中心差分)")
rng = np.random.default_rng(7)
n = 5
data = rng.standard_normal((1, n))
W_data = rng.standard_normal((n, 3))
v_data = rng.standard_normal((1, 3))
eps = 1e-6


def f_np(arr):
    """纯 NumPy float64 参考实现: L = sum(tanh(x @ W) * v)  (处处光滑, 无折点)"""
    return float((np.tanh(arr @ W_data) * v_data).sum())


t = Tensor(data.astype(np.float32), requires_grad=True)
h = Tanh()(t.matmul(Tensor(W_data.astype(np.float32))))
loss = (h * Tensor(v_data.astype(np.float32))).sum()
loss.backward()
print("loss = %.6f" % float(loss.data))
print("analytic grad :", np.round(t.grad.ravel(), 6))
num = np.zeros((1, n), dtype=np.float64)
for i in range(n):
    d = np.zeros((1, n))
    d[0, i] = eps
    lp = f_np(data + d)
    lm = f_np(data - d)
    num[0, i] = (lp - lm) / (2 * eps)
print("numeric grad :", np.round(num.ravel(), 6))
err = float(np.max(np.abs(t.grad.astype(np.float64) - num)))
print("max abs err  = %.3e  -> %s" % (err, "PASS" if err < 1e-6 else "FAIL"))
assert err < 1e-6

# ---------------------------------------------------------------- 3
head("[3] 广播梯度归约: x(4,3) + b(3,)")
x = Tensor(rng.standard_normal((4, 3)).astype(np.float32), requires_grad=True)
b = Tensor(rng.standard_normal((3,)).astype(np.float32), requires_grad=True)
y = x + b
seed = np.arange(12, dtype=np.float32).reshape(4, 3)   # 上游梯度 dL/dy
y.backward(seed)
print("上游梯度 dL/dy =")
print(seed)
print("x.shape=%s x.grad.shape=%s  (原样透传)" % (x.shape, x.grad.shape))
print("b.shape=%s b.grad.shape=%s  (沿 batch 维求和)" % (b.shape, b.grad.shape))
print("b.grad =", np.round(b.grad, 6), " == 每列求和", np.round(seed.sum(axis=0), 6))
assert b.grad.shape == (3,)
assert np.allclose(b.grad, seed.sum(axis=0), atol=1e-6)
assert np.allclose(x.grad, seed, atol=1e-6)

# ---------------------------------------------------------------- 4
head("[4] 菱形计算图: 同一叶子经两条路径汇聚")
x = Tensor([1.0, 2.0], requires_grad=True)
a = x * 2.0
b2 = x * 3.0
z = (a + b2).sum()
z.backward()
print("z = sum((x*2) + (x*3))  =>  dL/dx = 2 + 3 = 5, 实际:", x.grad)
assert np.allclose(x.grad, 5.0)

# ---------------------------------------------------------------- 5
head("[5] zero_grad / retain_graph 语义")
x = Tensor([2.0], requires_grad=True)
(x * 3).backward()
print("first backward  : x.grad =", float(x.grad[0]))
x.zero_grad()
print("after zero_grad : x.grad =", x.grad)
(x * 5).backward()
print("second backward : x.grad =", x.grad, " (梯度默认累加; zero_grad 后从 0 开始)")
x2 = Tensor([2.0], requires_grad=True)
y3 = x2 * 7
y3.backward(retain_graph=True)
print("retain_graph=True 后再次 backward:")
y3.backward()
print("x2.grad =", x2.grad, " (同一张图两次累加 7+7=14)")
assert np.allclose(x2.grad, 14.0)

# ---------------------------------------------------------------- 6
head("[6] 计算图可视化: z = x@W ; y = z + b ; L = (y @ c).sum()")
x = Tensor([[1.0, 2.0]], requires_grad=True)
W = Tensor([[0.5, -0.5, 1.0], [1.0, 2.0, -1.0]], requires_grad=True)
b = Tensor([0.1, 0.2, 0.3], requires_grad=True)
c = Tensor([[1.0], [2.0], [3.0]])          # 常量, requires_grad=False
z = x.matmul(W)
y = z + b
L = y.matmul(c).sum()
print("前向(建图): x%s --[matmul]--> z%s --[add b]--> y%s --[matmul c + sum]--> L=%s"
      % (x.shape, z.shape, y.shape, np.round(L.data, 6)))
print("_grad_fn 链: L(%s) <- y(%s) <- z(%s)"
      % (gfn(L), gfn(y), gfn(z)))
L.backward()
print("反向(种子 dL/dL=1):")
print("  x.grad =", np.round(x.grad, 6).tolist())
print("  W.grad =", np.round(W.grad, 6).tolist())
print("  b.grad =", np.round(b.grad, 6))
# 手算: L = y @ c = sum(y_i * c_i) => dL/db = c, dL/dW = x^T @ c^T, dL/dx = c^T @ W^T
db = c.data[:, 0]
dW = x.data.T @ c.data.T
dx = c.data.T @ W.data.T
print("手算对照: b", np.round(db, 6), " / W", np.round(dW, 6).tolist(), " / x", np.round(dx, 6).tolist())
assert np.allclose(b.grad, db, atol=1e-5)
assert np.allclose(W.grad, dW, atol=1e-5)
assert np.allclose(x.grad, dx, atol=1e-5)
print("三条梯度全部与手算一致 -> PASS")

head("[6b] 已知局限: 同一中间节点重复入图 (L = (y*y).sum())")
x = Tensor([[1.0, 2.0]], requires_grad=True)
W = Tensor([[0.5, -0.5, 1.0], [1.0, 2.0, -1.0]], requires_grad=True)
b = Tensor([0.1, 0.2, 0.3], requires_grad=True)
z = x.matmul(W)
y = z + b
L2 = (y * y).sum()
L2.backward()
y_np = x.data @ W.data + b.data
db_true = 2 * y_np
print("autograd b.grad :", np.round(b.grad, 6))
print("true    2*y     :", np.round(db_true, 6))
print("比值            :", np.round(b.grad / db_true, 4), " -> 第二路梯度被图释放吞掉")
xb = Tensor([3.0], requires_grad=True)
aa = xb * xb
cc = aa + aa
cc.backward()
print("最小复现 b = a + a 且 a = x*x : x.grad =", xb.grad, "(正确应为 12.0)")
print("原因: MulBackward/AddBackward 的 saved_tensors 里同一对象出现两次时,")
print("      第一次递归回传后 _grad_fn=None 释放了图, 第二条路径无法继续回传;")
print("      PyTorch 用拓扑排序 + 每节点梯度缓冲 + 消费者计数来正确累加所有路径。")

# ---------------------------------------------------------------- 7
head("[7] 训练闭环: 2 层网络拟合 XOR (手动 SGD)")
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32)
T = np.array([[0., 1.], [1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
Xt, Tt = Tensor(X), Tensor(T)
mse = MSELoss()
W1 = Tensor((rng.standard_normal((2, 8)) * 0.5).astype(np.float32), requires_grad=True)
c1 = Tensor(np.zeros((1, 8), dtype=np.float32), requires_grad=True)
W2 = Tensor((rng.standard_normal((8, 2)) * 0.5).astype(np.float32), requires_grad=True)
c2 = Tensor(np.zeros((1, 2), dtype=np.float32), requires_grad=True)
lr = 0.1
for step in range(1, 201):
    h = ReLU()(Xt.matmul(W1) + c1)
    o = h.matmul(W2) + c2
    loss = mse(o, Tt)          # Module 04 的 MSELoss, 由 enable_autograd 打补丁挂 MSEBackward
    for p in (W1, c1, W2, c2):
        p.zero_grad()
    loss.backward()
    for p in (W1, c1, W2, c2):
        p.data = (p.data - lr * p.grad).astype(np.float32)
    if step in (1, 20, 100, 200):
        print("step %3d  MSE = %.6f" % (step, float(loss.data)))
final_mse = float(mse(ReLU()(Xt.matmul(W1) + c1).matmul(W2) + c2, Tt).data)
print("final MSE = %.6f" % final_mse)
assert final_mse < 0.05, final_mse

print()
print("ALL DEMO CHECKS PASSED")

