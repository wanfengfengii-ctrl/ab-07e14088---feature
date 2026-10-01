# 超声探头延迟计划编译服务（Delay Plan Compiler）

把逐阵元标定出的整数目标延迟编译为探头固件可容纳的**少量整数斜坡**序列，
避免逐点取整产生超出硬件换挡能力的跳变。

## 问题定义

给定 `n`（12–48）个整数目标延迟，求整数延迟序列 `x[0..n-1]`，满足：

| 约束 | 说明 |
| --- | --- |
| 全局闭区间 | `minDelay <= x[i] <= maxDelay` |
| 锚点精确命中 | 2–8 个 `(element, delay)`，`x[element] == delay` |
| 相邻变化量 | `|x[i+1] - x[i]| <= maxStep` |
| 斜坡数上限 | 相邻差序列中"相等值的最大连续段数"（即恒定斜率段）`<= maxRamps` |
| 最短斜坡保持（可选） | 填写 `minRampSteps`（2–8）后，每段斜坡（**首段与末段同样如此**）都必须至少覆盖该数量的相邻步；换挡只能发生在前一斜率已经满足保持长度之后 |

> `minRampSteps` 缺省即旧行为：请求、响应、最优解与失败原因完全兼容。
> 启用后，求解器把保持长度与锚点、`maxStep`、斜坡数上限放在**同一张
> 动态规划**内联合裁决（状态增加"当前斜坡已持续步长"维度），绝不先求
> 无保持最优序列再合并或删除短段。

可行方案按以下顺序逐级最小化（全部使用整数裁决）：

1. **最大绝对误差** `max |x[i] - target[i]|`
2. **总绝对误差** `Σ |x[i] - target[i]|`
3. **实际斜坡数**
4. **延迟序列字典序**（最小者优先）

求解器为精确动态规划（状态：位置 × 取值 × 入边斜率 × 已用斜坡数；
启用 `minRampSteps` 时再追加"当前斜坡已持续步长"一维，饱和桶表示
"已满足保持长度"）：前向瓶颈 DP 求最小最大误差 `E*` → 在
`|x[i]-target[i]| <= E*` 收紧域上做后缀 DP 最小化组合成本
`总误差 × (斜坡上限+1) + 斜坡数`（字典序等价）→ 依据后缀层贪心重建
字典序最小序列。换坡转移只允许离开已饱和的斜坡，末层只接受饱和终态，
因此保持长度从一开始就与锚点、步长和斜坡数联合裁决。测试中以暴力枚举
对小规模随机实例逐一交叉验证最优性（含/不含保持长度两组参数化用例）。

## API

### `POST /api/delay-plans/compile`

请求体：

```json
{
  "targets":  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
               20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9],
  "minDelay": 0,
  "maxDelay": 40,
  "maxStep": 2,
  "maxRamps": 3,
  "minRampSteps": 3,
  "anchors": [
    {"element": 0,  "delay": 10},
    {"element": 12, "delay": 20},
    {"element": 23, "delay": 9}
  ]
}
```

字段约束：`targets` 12–48 个整数；`minDelay <= maxDelay`；`maxStep >= 0`；
`maxRamps >= 1`；`anchors` 2–8 个，`element` 为 0 基阵元下标且互不重复；
`minRampSteps` 可选，取值 2–8，缺省即旧行为。
数值包络：所有延迟/目标/步长绝对值 `<= 100000`，`maxRamps <= 47`。

可行响应（HTTP 200，`feasible: true`）：

```json
{
  "feasible": true,
  "delays":  [10, 11, "..."],
  "errors":  [0, 0, "..."],
  "maxAbsError": 0,
  "totalAbsError": 0,
  "rampCount": 2,
  "ramps": [
    {"startElement": 0,  "endElement": 11, "slope": 1,  "startValue": 10, "endValue": 21, "stepCount": 11},
    {"startElement": 11, "endElement": 23, "slope": -1, "startValue": 21, "endValue": 9,  "stepCount": 12}
  ]
}
```

- `errors[i] = delays[i] - targets[i]`（带符号逐阵元误差）；
- `ramps` 给出每段斜坡的起止阵元（含端点）、整数斜率与起止取值，
  各段首尾相接并覆盖全部阵元；
- **仅当**请求填写了 `minRampSteps` 时，每段斜坡额外给出
  `stepCount = endElement - startElement`（相邻步数，必 `>= minRampSteps`
  且与起止阵元一致）；未启用时该字段不出现，响应形状与旧版本完全一致。

不可行响应（HTTP 200，`feasible: false`）——**不输出任何部分延迟表**
（`delays`/`errors`/`ramps` 均为 `null`）：

```json
{
  "feasible": false,
  "reason": "anchor-step-conflict",
  "conflict": {
    "startElement": 1, "endElement": 5,
    "startDelay": 0,   "endDelay": 20,
    "requiredChange": 20, "allowedChange": 8, "maxStep": 2
  },
  "delays": null, "errors": null, "ramps": null,
  "maxAbsError": null, "totalAbsError": null, "rampCount": null
}
```

`reason` 取值：

| reason | 含义 |
| --- | --- |
| `anchor-out-of-bounds` | 锚点延迟超出全局闭区间 |
| `anchor-step-conflict` | 锚点间距内变化量超过 `maxStep` 换挡能力；`conflict` 给出冲突区间（确定性选择下标最小的冲突锚点对） |
| `ramp-limit-exceeded` | 斜坡数上限过紧（**未启用**保持长度时）；附 `minRampsRequired`（实际所需最少斜坡数） |
| `ramp-span-conflict` | **启用** `minRampSteps` 后联合约束无解（保持长度与锚点、`maxStep`、斜坡数上限共同裁决失败，首段/末段过短或换挡过快均在此列）；附 `minRampSteps`，不输出任何部分延迟、误差或斜坡 |

参数非法（数量/范围/重复锚点等）返回 HTTP 422；实例超出求解器资源
包络（DP 状态数超过 60M）同样返回 422 并附说明。

### `GET /healthz`

健康检查，返回 `{"status": "ok"}`。

## 构建与运行

```bash
# 构建镜像并以后台方式运行 API（默认端口 8000，可用 API_PORT 覆盖）
docker compose build
API_PORT=9000 docker compose up -d api

curl http://localhost:9000/healthz
```

Compose 中的 `api` 服务带健康检查（轮询 `/healthz`），端口映射为
`${API_PORT:-8000}:8000`。

## 一次性 verify 服务

```bash
docker compose up --exit-code-from verify
# 或显式：docker compose up verify --abort-on-container-exit --exit-code-from verify
```

`verify` 依赖 `api` 健康后启动，依次汇总并以**退出码**报告：

1. **代码测试**：镜像内运行 pytest 套件（含暴力枚举交叉验证）；
2. **镜像构建产物**：应用包与版本、依赖可导入、OpenAPI 模式可生成等；
3. **API 冒烟**：对在线服务发起典型编译请求（可行 / 锚点冲突 /
   斜坡超限 / 最短保持约束下可行 / 最短保持联合冲突），逐项复核响应契约。

全部通过退出码为 0，任一组失败退出码为 1。

## 本地开发

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/pytest                       # 运行测试
.venv/bin/uvicorn app.main:app --reload --port 8000
```

## 项目结构

```
app/
  main.py      FastAPI 路由（/healthz, /api/delay-plans/compile）
  schemas.py   请求校验（pydantic）
  solver.py    精确整数 DP 求解器
  verify.py    一次性校验入口（python -m app.verify）
tests/
  test_solver.py  求解器单测 + 暴力枚举交叉验证
  test_api.py     API 契约与 422 校验测试
Dockerfile          应用镜像
docker-compose.yml  api（健康检查、API_PORT）+ verify（一次性）
```
