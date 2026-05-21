from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the CARLA-LeWM interactive project dashboard HTML.")
    parser.add_argument("--output", type=Path, default=Path("interactive_plan.html"))
    return parser.parse_args()


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CARLA-LeWM 小规模长程驾驶执行看板</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --ink: #1d2433;
      --muted: #657082;
      --line: #d9dee7;
      --accent: #0b6bcb;
      --accent-2: #0f8a72;
      --warn: #b96b00;
      --bad: #b42318;
      --good-bg: #e9f7f2;
      --warn-bg: #fff4df;
      --bad-bg: #fdeceb;
      --shadow: 0 10px 28px rgba(21, 32, 54, 0.10);
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, "Segoe UI", Arial, sans-serif; background: var(--bg); color: var(--ink); line-height: 1.55; letter-spacing: 0; }
    header { background: #152033; color: #fff; padding: 22px 28px; }
    header h1 { margin: 0 0 6px; font-size: 24px; font-weight: 760; }
    header p { margin: 0; color: #cbd5e1; max-width: 980px; }
    .toolbar { position: sticky; top: 0; z-index: 20; display: flex; gap: 10px; align-items: center; padding: 10px 18px; background: rgba(255,255,255,.96); border-bottom: 1px solid var(--line); }
    .toolbar input { width: min(360px, 45vw); padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; font-size: 14px; }
    button, .tab { border: 1px solid var(--line); background: #fff; color: var(--ink); padding: 7px 10px; border-radius: 6px; cursor: pointer; font-size: 14px; }
    button:hover, .tab.active { border-color: var(--accent); color: var(--accent); }
    .layout { display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 18px; max-width: 1240px; margin: 0 auto; padding: 18px; }
    nav { position: sticky; top: 58px; align-self: start; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); padding: 12px; }
    nav a { display: block; color: var(--ink); text-decoration: none; padding: 7px 8px; border-radius: 6px; font-size: 14px; }
    nav a:hover { background: #eef5ff; color: var(--accent); }
    main { display: grid; gap: 16px; }
    section, details { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); padding: 16px; }
    section h2, details h2 { margin: 0 0 10px; font-size: 18px; }
    details summary { cursor: pointer; font-weight: 720; font-size: 17px; }
    a { color: var(--accent); }
    .metrics { display: grid; grid-template-columns: repeat(5, minmax(130px, 1fr)); gap: 10px; }
    .metric { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfcfe; }
    .metric strong { display: block; font-size: 18px; }
    .metric span { color: var(--muted); font-size: 13px; }
    .callout { border-left: 4px solid var(--warn); background: var(--warn-bg); padding: 10px 12px; border-radius: 6px; margin-top: 12px; }
    .ok { border-left-color: var(--accent-2); background: var(--good-bg); }
    .bad { border-left-color: var(--bad); background: var(--bad-bg); }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; border-bottom: 1px solid var(--line); padding: 8px; vertical-align: top; }
    th { background: #f1f4f8; }
    code { background: #eef2f7; padding: 1px 5px; border-radius: 4px; }
    pre { overflow: auto; background: #101827; color: #d9e7ff; padding: 12px; border-radius: 8px; }
    .tag { display: inline-block; border-radius: 999px; padding: 2px 8px; font-size: 12px; border: 1px solid var(--line); color: var(--muted); }
    .tag.good { color: #0f6b50; border-color: #b7e4d4; background: #e9f7f2; }
    .tag.warn { color: #8a4d00; border-color: #f1cf8a; background: #fff4df; }
    .tag.bad { color: #9f241a; border-color: #f2b8b5; background: #fdeceb; }
    .bar { height: 10px; background: #e7edf5; border-radius: 999px; overflow: hidden; min-width: 130px; }
    .bar > span { display: block; height: 100%; background: var(--accent); border-radius: 999px; }
    .bar.bad > span { background: var(--bad); }
    mark.search-hit { background: #fff08a; padding: 0 2px; }
    mark.current-hit { outline: 2px solid #f59e0b; }
    .hidden-kind { display: none; }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      nav { position: static; }
      .metrics, .grid { grid-template-columns: 1fr; }
      .toolbar { flex-wrap: wrap; }
      .toolbar input { width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <h1>CARLA-LeWM 小规模长程驾驶执行看板</h1>
    <p>先看当前结论，再看数据、W&B、闭环评估证据；所有 run name 旁边都解释了配置含义。</p>
  </header>
  <div class="toolbar">
    <input id="search" placeholder="搜索 phase / W&B / tiny / small / IFD / QC">
    <button id="prev" type="button">&lt;</button>
    <button id="next" type="button">&gt;</button>
    <span id="count" class="tag">0</span>
    <button class="tab active" data-filter="all" type="button">全部</button>
    <button class="tab" data-filter="data" type="button">数据</button>
    <button class="tab" data-filter="train" type="button">训练</button>
    <button class="tab" data-filter="eval" type="button">评估</button>
    <button class="tab" data-filter="risk" type="button">风险</button>
  </div>
  <div class="layout">
    <nav>
      <a href="#verdict">当前结论</a>
      <a href="#terms">名词速查</a>
      <a href="#decoder">实验名解码</a>
      <a href="#data">数据证据</a>
      <a href="#wandb">W&B 证据</a>
      <a href="#eval">闭环评估</a>
      <a href="#phases">阶段计划</a>
      <a href="#commands">命令</a>
      <a href="#gates">Go / No-Go</a>
    </nav>
    <main id="content">
      <section id="verdict" data-kind="all">
        <h2>当前结论</h2>
        <div class="metrics">
          <div class="metric"><span>Phase</span><strong>D0 pass</strong><span>采集、QC、训练、短评估完成</span></div>
          <div class="metric"><span>Dataset</span><strong>48k</strong><span>D0-train 严格 QC 通过</span></div>
          <div class="metric"><span>Best offline</span><strong>small</strong><span>test/loss 1.1324</span></div>
          <div class="metric"><span>Best closed-loop</span><strong>150m</strong><span>tiny/small throttle-only 通过</span></div>
          <div class="metric"><span>Next</span><strong>steering</strong><span>修动作目标和转向规划</span></div>
        </div>
        <div class="callout ok">
          当前状态：数据和训练流水线可信；在隔离 CARLA 2100 端口和 throttle-only 简化目标下，tiny/small 都能 150m 无 hard infraction。200m 边界约 191m，下一步是转向/action objective 校准。
        </div>
      </section>

      <section id="terms" data-kind="all">
        <h2>名词速查</h2>
        <table>
          <tr><th>Term</th><th>解释</th></tr>
          <tr><td><code>CARLA</code></td><td>自动驾驶仿真器，本项目使用它生成相机、车辆控制、车道、碰撞和红灯事件。</td></tr>
          <tr><td><code>LeWM</code></td><td>LeWorldModel，使用视觉输入学习 latent dynamics 的小世界模型。</td></tr>
          <tr><td><code>ViT tiny / small</code></td><td>视觉 Transformer 编码器大小；本轮 tiny 和 small 都训练并做了短程闭环评估。</td></tr>
          <tr><td><code>W&B</code></td><td>Weights & Biases，用于记录训练曲线、配置、summary 和 run 链接。</td></tr>
          <tr><td><code>QC</code></td><td>Quality Control，逐帧检查缺帧、空白、动作异常、碰撞、离路、红灯和 blocked。</td></tr>
          <tr><td><code>IFD</code></td><td>Infraction-Free Distance，首次 hard infraction 前的行驶距离，越高越好。</td></tr>
          <tr><td><code>Mini Driving Score</code></td><td>路线完成度乘违规惩罚，用来防止只看是否开到终点。</td></tr>
          <tr><td><code>CEM64</code></td><td>Cross-Entropy Method 规划器，64 个动作样本、2 次迭代、horizon 4，用于短程 sanity eval。</td></tr>
          <tr><td><code>throttle-only</code></td><td>当前有效简化目标：只允许模型选择油门，steer/brake 固定为 0，用来验证最小闭环能力。</td></tr>
          <tr><td><code>sim_delta_s</code></td><td>相邻控制 tick 的 CARLA 仿真时间差；过大说明有外部客户端推进世界，评估应判无效。</td></tr>
          <tr><td><code>D0</code></td><td>单车、白天、固定简化路线、强约束质量门控，是第一阶段最简单数据分布。</td></tr>
        </table>
      </section>

      <section id="decoder" data-kind="train">
        <h2>实验名解码</h2>
        <p><code>d0_tiny_h3_fs5_fast_e5</code>：D0 简化路线、ViT tiny、3 帧历史、5 帧合并一个动作步、fast HDF5、训练 5 epoch。</p>
        <table>
          <tr><th>片段</th><th>含义</th></tr>
          <tr><td><code>d0</code></td><td>单车低速白天简化场景。</td></tr>
          <tr><td><code>tiny / small</code></td><td>ViT encoder 大小。</td></tr>
          <tr><td><code>h3</code></td><td>模型输入 3 个历史视觉状态。</td></tr>
          <tr><td><code>fs5</code></td><td>CARLA 20Hz 控制每 5 帧合并，模型步长约 0.25s。</td></tr>
          <tr><td><code>fast</code></td><td>使用未压缩/chunked HDF5，避免训练随机读取卡住。</td></tr>
          <tr><td><code>e5</code></td><td>训练 5 epoch，用于第一轮可比实验。</td></tr>
        </table>
      </section>

      <section id="data" data-kind="data">
        <h2>数据证据</h2>
        <table>
          <tr><th>Dataset</th><th>Episodes</th><th>Frames</th><th>QC</th><th>Hard infractions</th></tr>
          <tr><td><code>D0-smoke</code></td><td>20</td><td>12,000</td><td><span class="tag good">pass</span></td><td>0 collision / 0 off-road / 0 red / 0 blocked</td></tr>
          <tr><td><code>D0-train</code></td><td>80</td><td>48,000</td><td><span class="tag good">pass</span></td><td>0 collision / 0 off-road / 0 red / 0 blocked</td></tr>
        </table>
        <div class="callout ok">读法：这张表证明 D0 训练数据本身足够干净，可以进入模型训练。它不能证明模型闭环会开车。</div>
      </section>

      <section id="wandb" data-kind="train">
        <h2>W&B 证据</h2>
        <table>
          <tr><th>Run</th><th>模型</th><th>Batch</th><th>Val/Test</th><th>曲线读法</th></tr>
          <tr>
            <td><a href="https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_tiny_h3_fs5_fast_e5-20260522-014214-db1cb3e1"><code>d0_tiny_h3_fs5_fast_e5</code></a><br>tiny、D0、fast HDF5、5 epoch。</td>
            <td>ViT tiny</td><td>192</td><td>val 2.9568<br>test 2.9620<br>pred 0.7015</td><td><div class="bar"><span style="width: 38%"></span></div>离线 loss 有下降，但闭环仍离路。</td>
          </tr>
          <tr>
            <td><a href="https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d0_small_h3_fs5_fast_e5-20260522-024715-2f674d5e"><code>d0_small_h3_fs5_fast_e5</code></a><br>small、D0、fast HDF5、5 epoch。</td>
            <td>ViT small</td><td>128</td><td>val 1.1281<br>test 1.1324<br>pred 0.5269</td><td><div class="bar"><span style="width: 75%"></span></div>离线 loss 更好，但闭环直接 blocked。</td>
          </tr>
        </table>
        <div class="callout">当前结论：W&B 曲线和 test loss 只能证明离线拟合变好。small loss 更低，但没有把 throttle-only 200m 边界推过 200m。</div>
      </section>

      <section id="eval" data-kind="eval">
        <h2>闭环评估</h2>
        <table>
          <tr><th>Policy</th><th>配置</th><th>IFD</th><th>Mini Score</th><th>结论</th></tr>
          <tr><td>Autopilot baseline</td><td>2 episodes, 100m cap</td><td>100.00m</td><td>100.0</td><td><span class="tag good">pass</span> 环境和指标可信。</td></tr>
          <tr><td>tiny checkpoint</td><td>150m cap, isolated port 2100, throttle-only</td><td>150.00m</td><td>100.0</td><td><span class="tag good">pass</span> 当前最小有效闭环结果。</td></tr>
          <tr><td>small checkpoint</td><td>150m cap, isolated port 2100, throttle-only</td><td>150.00m</td><td>100.0</td><td><span class="tag good">pass</span> 和 tiny 持平。</td></tr>
          <tr><td>tiny checkpoint</td><td>200m cap, isolated port 2100, throttle-only</td><td>191.11m</td><td>70.0</td><td><span class="tag warn">boundary</span> 近终点 off-road。</td></tr>
          <tr><td>small checkpoint</td><td>200m cap, isolated port 2100, throttle-only</td><td>191.87m</td><td>70.0</td><td><span class="tag warn">boundary</span> 更大 ViT 没有解决边界。</td></tr>
        </table>
        <div class="callout">有效性说明：早期 port 2000 model eval 被外部 HIL client tick 污染；当前 evaluator 已写 action/timing trace，并在 `sim_delta_s` 过大时 fail。</div>
      </section>

      <section id="phases" data-kind="all">
        <h2>阶段计划</h2>
        <details open data-kind="data"><summary>Phase 0-1：环境、采集、逐帧 QC <span class="tag good">done</span></summary>
          <p>目标：建立 GitHub repo、CARLA 0.9.16 采集、逐帧 QC 和 contact sheet 人工核查。</p>
          <p>结果：D0-smoke 和 D0-train 都 strict pass；40s 版本因长尾离路/碰撞被放弃，30s 简化数据进入训练。</p>
        </details>
        <details open data-kind="train"><summary>Phase 2-3：tiny / small 训练 <span class="tag good">done</span></summary>
          <p>目标：先 tiny，再在 tiny 闭环不理想后尝试 small；全程 W&B 记录。</p>
          <p>结果：small 离线 loss 更低，但闭环更差；模型大小不是当前主瓶颈。</p>
        </details>
        <details open data-kind="eval"><summary>Phase 4：简化闭环 sanity <span class="tag good">150m pass</span></summary>
          <p>目标：用 IFD 和 Mini Driving Score 检查模型能否短程不出错。</p>
          <p>结果：隔离 CARLA + throttle-only 下 tiny/small 均 150m pass；200m 在约 191m off-road。下一步需要恢复安全 steering。</p>
        </details>
        <details data-kind="risk"><summary>Phase 5：长程 500m / D1 低密交通 <span class="tag">not ready</span></summary>
          <p>进入条件：200m throttle-only pass，并且 steering-enabled 目标至少 150m IFD 且无 hard infraction。</p>
        </details>
      </section>

      <section id="commands" data-kind="all">
        <h2>关键命令</h2>
        <pre><code class="language-bash">python -m carla_lewm_drive.dataset_qc.export_fast_hdf5 \
  --src data/d0_train/carla_d0_train.h5 \
  --dst data/d0_train/carla_d0_train_fast.h5 \
  --chunk-frames 256 --overwrite

python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_tiny.yaml \
  --dataset-path data/d0_train/carla_d0_train_fast.h5 \
  --batch-size 192 --num-workers 2 --max-epochs 5 \
  --output-dir outputs/d0_tiny_h3_fs5_fast_e5 \
  --run-name d0_tiny_h3_fs5_fast_e5

python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d0_throttle_only.yaml \
  --checkpoint outputs/d0_tiny_h3_fs5_fast_e5/best.pt \
  --output-dir outputs/d0_eval_tiny_throttle_only_150m</code></pre>
      </section>

      <section id="gates" data-kind="risk">
        <h2>Go / No-Go</h2>
        <div class="grid">
          <div class="callout ok"><strong>Continue</strong><br>修 delta-progress objective、pred-aux head、steering-safe prior，然后复跑 150m/200m。</div>
          <div class="callout"><strong>Pause</strong><br>离线 loss 继续下降但 IFD 不提升时，优先诊断控制闭环，不扩大模型。</div>
          <div class="callout bad"><strong>Stop</strong><br>W&B 未启动、QC 非严格通过、autopilot baseline 失败、或 `sim_delta_s` 出现外部 tick 跳变。</div>
          <div class="callout"><strong>Scale</strong><br>只有 steering-enabled 目标呈 capacity-limited 时，才继续跑更大 ViT。</div>
        </div>
      </section>
    </main>
  </div>
  <script>
    const content = document.getElementById('content');
    const search = document.getElementById('search');
    const count = document.getElementById('count');
    let hits = [];
    let current = -1;

    function clearMarks() {
      content.querySelectorAll('mark.search-hit').forEach(mark => {
        const text = document.createTextNode(mark.textContent);
        mark.replaceWith(text);
      });
      content.normalize();
    }
    function markText(node, query) {
      const text = node.nodeValue;
      const lower = text.toLowerCase();
      const q = query.toLowerCase();
      let index = lower.indexOf(q);
      if (index < 0) return;
      const frag = document.createDocumentFragment();
      let last = 0;
      while (index >= 0) {
        frag.appendChild(document.createTextNode(text.slice(last, index)));
        const mark = document.createElement('mark');
        mark.className = 'search-hit';
        mark.textContent = text.slice(index, index + query.length);
        frag.appendChild(mark);
        last = index + query.length;
        index = lower.indexOf(q, last);
      }
      frag.appendChild(document.createTextNode(text.slice(last)));
      node.replaceWith(frag);
    }
    function doSearch() {
      clearMarks();
      const q = search.value.trim();
      hits = [];
      current = -1;
      if (!q) { count.textContent = '0'; return; }
      const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
          if (!node.nodeValue.toLowerCase().includes(q.toLowerCase())) return NodeFilter.FILTER_REJECT;
          if (node.parentElement && ['SCRIPT', 'STYLE', 'MARK'].includes(node.parentElement.tagName)) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }
      });
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      nodes.forEach(node => markText(node, q));
      hits = Array.from(content.querySelectorAll('mark.search-hit'));
      count.textContent = String(hits.length);
      jump(0);
    }
    function jump(delta) {
      if (!hits.length) return;
      if (current >= 0) hits[current].classList.remove('current-hit');
      current = (current + delta + hits.length) % hits.length;
      hits[current].classList.add('current-hit');
      hits[current].scrollIntoView({behavior: 'smooth', block: 'center'});
    }
    let timer = null;
    search.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(doSearch, 100); });
    document.getElementById('prev').addEventListener('click', () => jump(-1));
    document.getElementById('next').addEventListener('click', () => jump(1));
    document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const filter = tab.dataset.filter;
      document.querySelectorAll('[data-kind]').forEach(el => {
        const kind = el.dataset.kind;
        el.classList.toggle('hidden-kind', filter !== 'all' && kind !== filter && kind !== 'all');
      });
    }));
  </script>
</body>
</html>
"""


def build(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML, encoding="utf-8")
    return output


def main() -> None:
    args = parse_args()
    print(build(args.output))


if __name__ == "__main__":
    main()
