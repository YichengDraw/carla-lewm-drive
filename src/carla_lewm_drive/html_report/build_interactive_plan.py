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
  <title>CARLA-LeWM 1km 城市驾驶报告</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f2e8;
      --panel: #fffaf0;
      --panel-strong: #fff3d6;
      --ink: #14233a;
      --muted: #5f6f82;
      --line: #e0d3bc;
      --accent: #1e5aa8;
      --good: #0c7a55;
      --warn: #b96b00;
      --bad: #a93b32;
      --good-bg: #e9f7f2;
      --warn-bg: #fff4df;
      --bad-bg: #fdeceb;
      --shadow: 0 10px 28px rgba(21, 32, 54, 0.10);
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Inter, "Segoe UI", Arial, sans-serif; color: var(--ink); background: var(--bg); line-height: 1.55; letter-spacing: 0; }
    header { background: #14233a; color: #fffaf0; padding: 22px 28px; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    h2 { margin: 0 0 12px; font-size: 18px; }
    h3 { margin: 14px 0 8px; font-size: 15px; }
    p { margin: 8px 0; }
    a { color: var(--accent); }
    code { background: #eef2f7; border-radius: 4px; padding: 1px 5px; }
    pre { margin: 8px 0; padding: 12px; border-radius: 8px; overflow: auto; background: #111927; color: #dbeafe; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; vertical-align: top; padding: 8px; border-bottom: 1px solid var(--line); }
    th { background: var(--panel-strong); }
    .toolbar { position: sticky; top: 0; z-index: 10; display: flex; gap: 10px; flex-wrap: wrap; align-items: center; padding: 10px 18px; background: rgba(255,250,240,.96); border-bottom: 1px solid var(--line); }
    .toolbar input { width: min(420px, 56vw); padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; font-size: 14px; }
    button, .tab { border: 1px solid var(--line); border-radius: 6px; padding: 7px 10px; background: #fff; color: var(--ink); cursor: pointer; }
    button:hover, .tab.active { border-color: var(--accent); color: var(--accent); }
    .layout { display: grid; grid-template-columns: 230px minmax(0, 1fr); gap: 18px; max-width: 1260px; margin: 0 auto; padding: 18px; }
    nav { position: sticky; top: 58px; align-self: start; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); box-shadow: var(--shadow); }
    nav a { display: block; padding: 7px 8px; border-radius: 6px; color: var(--ink); text-decoration: none; font-size: 14px; }
    nav a:hover { color: var(--accent); background: #eef5ff; }
    main { display: grid; gap: 16px; }
    section, details { padding: 16px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); box-shadow: var(--shadow); }
    summary { cursor: pointer; font-weight: 720; }
    .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 10px; }
    .metric { min-height: 92px; border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fffdf7; }
    .metric strong { display: block; margin: 2px 0; font-size: 20px; }
    .metric span { color: var(--muted); font-size: 13px; }
    .tag { display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 2px 8px; color: var(--muted); font-size: 12px; white-space: nowrap; }
    .good { color: var(--good); border-color: #b7e4d4; background: var(--good-bg); }
    .warn { color: var(--warn); border-color: #f1cf8a; background: var(--warn-bg); }
    .bad { color: var(--bad); border-color: #f2b8b5; background: var(--bad-bg); }
    .callout { border-left: 4px solid var(--warn); background: var(--warn-bg); padding: 10px 12px; border-radius: 6px; margin-top: 12px; }
    .callout.good { border-left-color: var(--good); background: var(--good-bg); }
    .callout.bad { border-left-color: var(--bad); background: var(--bad-bg); }
    .hidden-kind { display: none; }
    .search-highlight { background: #fff08a; padding: 0 2px; }
    .search-highlight.active { outline: 2px solid #c5652f; }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      nav { position: static; }
      .toolbar input { width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <h1>CARLA-LeWM 1km 城市驾驶报告</h1>
    <p>当前硬目标：在简化城市街道中，无车无人，遵守红绿灯，不撞车道线、不离路、不碰撞、不阻塞，至少行驶 1km。</p>
  </header>
  <div class="toolbar">
    <input id="keywordSearch" placeholder="搜索 ft46 / hybrid / pure / redlight / W&B / 1km">
    <button id="prevSearch" type="button">&lt;</button>
    <button id="nextSearch" type="button">&gt;</button>
    <span id="searchStatus" class="tag">0</span>
    <button class="tab active" data-filter="all" type="button">全部</button>
    <button class="tab" data-filter="task" type="button">任务</button>
    <button class="tab" data-filter="train" type="button">训练</button>
    <button class="tab" data-filter="eval" type="button">评估</button>
    <button class="tab" data-filter="risk" type="button">风险</button>
  </div>
  <div class="layout">
    <nav>
      <a href="#verdict">当前结论</a>
      <a href="#task">任务定义</a>
      <a href="#evidence">证据链</a>
      <a href="#wandb">W&B</a>
      <a href="#commands">复现实验</a>
      <a href="#next">后续计划</a>
      <a href="#terms">名词速查</a>
    </nav>
    <main id="content">
      <section id="verdict" data-kind="all">
        <h2>当前结论</h2>
        <div class="metrics">
          <div class="metric"><span>目标状态</span><strong>达成</strong><span>hybrid policy 红灯规则下 1km clean</span></div>
          <div class="metric"><span>最终距离</span><strong>1000m</strong><span>route_complete，Infraction-Free Distance 1000m</span></div>
          <div class="metric"><span>Mini Score</span><strong>100.0</strong><span>forced-green 与 redlight 均满分</span></div>
          <div class="metric"><span>安全违规</span><strong>0</strong><span>lane/offroad/collision/blocked/red/speed 全 0</span></div>
          <div class="metric"><span>纯 action</span><strong>459m</strong><span>FT46 last 仍在 lane invasion 失败</span></div>
          <div class="metric"><span>红灯停车</span><strong>7256</strong><span>red_light_stop 帧，14 段停车，无闯红灯</span></div>
        </div>
        <div class="callout good">
          结论：纯 FT46 ViT action policy 还没有学稳长时程闭环，best/last 都在 456-459m 左右车道侵入。把 FT46 action 限制为 residual，并从第一帧开始用 semantic-geometry controller 提供 85% steer 后，模型在 route6 上完成 forced-green 1km；同一 hybrid 在真实红灯监控下也完成 1km，期间触发 7256 帧红灯停车且 0 次红灯违规。
        </div>
        <div class="callout bad">
          重要边界：当前 1km 结果不能写成“纯 LeWM 已经会开 1km”。它证明的是任务、数据、红灯评估和 residual/hybrid 安全闭环已经跑通；纯 LeWM action-only 仍需要单独通过 1km gate。
        </div>
      </section>

      <section id="task" data-kind="task">
        <h2>任务定义</h2>
        <table>
          <tr><th>项目</th><th>当前设置</th><th>通过标准</th></tr>
          <tr><td>场景</td><td>CARLA Town03，ClearNoon，route spawn 6，无其他车辆/行人。</td><td>先证明单路线可学，后续再扩路线、天气和交通。</td></tr>
          <tr><td>主指标</td><td><code>Infraction-Free Distance</code> 与 <code>Mini Driving Score</code>。</td><td>1km 内 lane invasion、offroad、collision、blocked、red-light、speed-limit 计数均为 0。</td></tr>
          <tr><td>最终策略</td><td><code>model_action_steer_speed_keep_smooth</code> + <code>semantic_geometry_guard</code> always-on。</td><td>steer = 0.15 * LeWM action + 0.85 * semantic geometry lane controller。</td></tr>
          <tr><td>红绿灯</td><td><code>force_green_lights=false</code>，<code>red_light_stop=true</code>，<code>stop_on_red_light=true</code>。</td><td>遇红灯停车，不发生红灯违规；阻塞检测在红灯停车段忽略。</td></tr>
        </table>
      </section>

      <section id="evidence" data-kind="eval">
        <h2>证据链</h2>
        <table>
          <tr><th>实验</th><th>结果</th><th>解释</th></tr>
          <tr><td><code>RF semantic_geometry baseline</code></td><td>forced-green 1km 满分；redlight 1km 满分。</td><td>语义相机几何特征足以解决这个简化驾驶任务，场景本身可达。</td></tr>
          <tr><td><code>FT44 semantic action distill</code></td><td>best 403.53m，last 399.03m，均 lane invasion。</td><td>右侧偏移 recovery 不足，模型持续向错误方向推。</td></tr>
          <tr><td><code>FT45 + right-tail DAgger</code></td><td>best 387.46m，last 455.72m，均 lane invasion。</td><td>右侧问题缓解后，左侧偏移又变成主失败。</td></tr>
          <tr><td><code>FT45 guarded</code></td><td>704.23m offroad。</td><td>偏离后才接管太晚；进入 OOD 后 RF 几何估计也会失效。</td></tr>
          <tr><td><code>FT46 balanced tail</code></td><td>offline test/action_steer_loss 0.0160；best 456.32m，last 458.99m。</td><td>离线 action loss 明显改善，但纯闭环仍发生符号/幅度漂移。</td></tr>
          <tr><td><code>FT46 always-on hybrid</code></td><td>forced-green 1000m，score 100，全部违规计数 0。</td><td>从第一帧约束 action residual，避免等偏移严重后再修。</td></tr>
          <tr><td><code>FT46 always-on hybrid redlight</code></td><td>redlight 1000m，score 100，red_light_count 0，red_light_stop_frames 7256。</td><td>实际经历红灯停车段，不是空跑绿灯路线。</td></tr>
        </table>
      </section>

      <section id="wandb" data-kind="train">
        <h2>W&B 与本地证据</h2>
        <table>
          <tr><th>Run / Artifact</th><th>用途</th><th>状态</th></tr>
          <tr>
            <td><a href="https://wandb.ai/yicheng132024-southern-university-of-science-technology/carla-lewm-drive/runs/d1_tiny_h1_fs1_route6_semantic_action_semgeom_distill_ft46_balanced_tail_1400-20260527-053855-13cc8c8c"><code>ft46 balanced tail 1400</code></a></td>
            <td>从 FT45 last 初始化，加入 FT44 右尾部和 FT45 左尾部 DAgger 数据，batch 192，W&B 从启动开始记录。</td>
            <td><span class="tag good">complete</span> best step 1200，test/action_steer_loss 0.0160。</td>
          </tr>
          <tr><td><code>outputs/d1_eval_ft46last_semantic_action_semgeom_slow_smooth_route6_1km_20260527_055133</code></td><td>FT46 last 纯 action forced-green。</td><td><span class="tag bad">458.99m</span> lane invasion。</td></tr>
          <tr><td><code>outputs/d1_eval_ft46best_semantic_action_semgeom_slow_smooth_route6_1km_20260527_055536</code></td><td>FT46 best 纯 action forced-green。</td><td><span class="tag bad">456.32m</span> lane invasion。</td></tr>
          <tr><td><code>outputs/d1_eval_ft46last_alwayson_hybrid_route6_1km_20260527_060026</code></td><td>FT46 last always-on hybrid forced-green。</td><td><span class="tag good">1000m</span> route_complete，score 100。</td></tr>
          <tr><td><code>outputs/d1_eval_ft46last_alwayson_hybrid_redlight_route6_1km_20260527_061347</code></td><td>FT46 last always-on hybrid redlight。</td><td><span class="tag good">1000m</span> route_complete，score 100，red_light_stop_frames 7256。</td></tr>
        </table>
      </section>

      <section id="commands" data-kind="eval">
        <h2>复现实验</h2>
        <h3>Forced-green hybrid 1km</h3>
        <pre><code>PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d1_route6_semantic_action_semgeom_distill_ft46_alwayson_hybrid_slow_smooth_tick_1km_port2110.yaml \
  --checkpoint outputs/d1_tiny_h1_fs1_route6_semantic_action_semgeom_distill_ft46_balanced_tail_1400/last.pt \
  --output-dir outputs/d1_eval_ft46last_alwayson_hybrid_route6_1km</code></pre>
        <h3>Redlight hybrid 1km</h3>
        <pre><code>PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.closed_loop_eval.evaluate \
  --config configs/eval_d1_route6_semantic_action_semgeom_distill_ft46_alwayson_hybrid_redlight_slow_smooth_tick_1km_port2110.yaml \
  --checkpoint outputs/d1_tiny_h1_fs1_route6_semantic_action_semgeom_distill_ft46_balanced_tail_1400/last.pt \
  --output-dir outputs/d1_eval_ft46last_alwayson_hybrid_redlight_route6_1km</code></pre>
        <h3>FT46 training</h3>
        <pre><code>PYTHONPATH=src .venv/bin/python -m carla_lewm_drive.driving_lewm.train \
  --config configs/train_d1_tiny_route6_semantic_action_semgeom_distill_ft46_balanced_tail_1400.yaml</code></pre>
      </section>

      <section id="next" data-kind="risk">
        <h2>后续计划</h2>
        <table>
          <tr><th>方向</th><th>原因</th><th>下一步</th></tr>
          <tr><td>从 hybrid 走 residual training</td><td>当前成功来自 85% geometry controller；纯 action 仍不稳定。</td><td>训练模型预测 RF controller residual 或校正项，并用闭环距离选 checkpoint。</td></tr>
          <tr><td>多路线验证</td><td>单 route6 达标还不足以证明泛化。</td><td>扩到 3/4/6/10/12，保持无车无天气变化。</td></tr>
          <tr><td>多天气混训</td><td>用户后续提到鲁棒性；当前只在 ClearNoon。</td><td>单路线多次稳定后，再加 Cloudy/Wet/SoftRain。</td></tr>
          <tr><td>更大 ViT</td><td>只有当 tiny residual 仍无法提高 LeWM 占比时才值得。</td><td>比较 0.85、0.7、0.5 blend 下的最远安全距离。</td></tr>
        </table>
      </section>

      <details id="pure-gate" data-kind="risk" open>
        <summary>纯 LeWM 验收门槛</summary>
        <table>
          <tr><th>Gate</th><th>标准</th><th>当前状态</th></tr>
          <tr><td>Pure forced-green 1km</td><td><code>semantic_geometry_guard.enabled=false</code>，不混入 RF/规则 steering。</td><td><span class="tag bad">未通过</span> FT46 best/last 在 456-459m lane invasion。</td></tr>
          <tr><td>Pure redlight 1km</td><td>在 pure forced-green 通过后再打开红灯停车与违规监控。</td><td><span class="tag warn">未开始</span> 等待 pure forced-green 先过。</td></tr>
          <tr><td>Blend annealing</td><td>hybrid 权重从 85% 降到 70%、50%、25%、0%，每档都记录最远无违规距离。</td><td><span class="tag warn">下一轮</span> 用它判断 LeWM action 真实贡献。</td></tr>
          <tr><td>Failure recovery audit</td><td>对 450m 左右失败区间画 <code>lane_offset</code>、<code>heading_error</code>、<code>model_steer</code> 与 teacher 差值。</td><td><span class="tag warn">下一轮</span> 验证是否是反馈增益不足或符号错误。</td></tr>
        </table>
      </details>

      <section id="terms" data-kind="all">
        <h2>名词速查</h2>
        <table>
          <tr><th>Term</th><th>解释</th></tr>
          <tr><td><code>LeWM</code></td><td>latent world model，用图像 latent 和动作预测未来 latent；本项目另加 action readout 做闭环控制。</td></tr>
          <tr><td><code>pure action</code></td><td>闭环控制只使用模型输出的 throttle/steer/brake，不使用 RF/geometry controller 纠偏。</td></tr>
          <tr><td><code>residual / hybrid</code></td><td>模型 action 与传统/几何控制器混合；当前成功配置是 15% LeWM steer + 85% semantic geometry steer。</td></tr>
          <tr><td><code>closed-loop</code></td><td>每一步重新根据当前观测输出动作；它保证有反馈形式，但不自动保证稳定性。</td></tr>
          <tr><td><code>semantic geometry</code></td><td>从语义分割相机图像中抽取车道/道路几何特征，再用轻量模型预测 lane offset 和 heading error。</td></tr>
          <tr><td><code>always-on hybrid</code></td><td>每一帧都融合模型 action 与 geometry controller，而不是等车辆偏离后再接管。</td></tr>
          <tr><td><code>Infraction-Free Distance</code></td><td>首次安全或规则违规前的行驶距离，当前任务的主指标。</td></tr>
          <tr><td><code>red_light_stop_frames</code></td><td>控制器主动因红灯刹停的帧数，用来确认 redlight eval 真的遇到红灯。</td></tr>
        </table>
      </section>
    </main>
  </div>

  <script>
    const input = document.querySelector('#keywordSearch');
    const count = document.querySelector('#searchStatus');
    const prev = document.querySelector('#prevSearch');
    const next = document.querySelector('#nextSearch');
    const tabs = [...document.querySelectorAll('.tab')];
    const sections = [...document.querySelectorAll('[data-kind]')];
    let marks = [];
    let current = -1;

    function clearMarks() {
      marks.forEach(mark => mark.replaceWith(document.createTextNode(mark.textContent)));
      marks = [];
      current = -1;
      count.textContent = '0';
    }

    function normalizeTextNode(node, query) {
      const text = node.nodeValue;
      const lower = text.toLowerCase();
      let idx = lower.indexOf(query);
      if (idx < 0) return;
      let cursor = 0;
      const frag = document.createDocumentFragment();
      while (idx >= 0) {
        frag.append(document.createTextNode(text.slice(cursor, idx)));
        const mark = document.createElement('mark');
        mark.className = 'search-highlight';
        mark.textContent = text.slice(idx, idx + query.length);
        frag.append(mark);
        marks.push(mark);
        cursor = idx + query.length;
        idx = lower.indexOf(query, cursor);
      }
      frag.append(document.createTextNode(text.slice(cursor)));
      node.replaceWith(frag);
    }

    function search() {
      clearMarks();
      const query = input.value.trim().toLowerCase();
      if (!query) return;
      const walker = document.createTreeWalker(document.querySelector('#content'), NodeFilter.SHOW_TEXT, {
        acceptNode(node) {
          if (!node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
          const parent = node.parentElement;
          if (!parent || ['SCRIPT', 'STYLE', 'MARK'].includes(parent.tagName)) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_ACCEPT;
        }
      });
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      nodes.forEach(node => normalizeTextNode(node, query));
      count.textContent = String(marks.length);
      if (marks.length) move(0);
    }

    function move(i) {
      if (!marks.length) return;
      if (current >= 0) marks[current].classList.remove('active');
      current = (i + marks.length) % marks.length;
      marks[current].classList.add('active');
      marks[current].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function filter(kind) {
      sections.forEach(section => {
        section.classList.toggle('hidden-kind', kind !== 'all' && section.dataset.kind !== kind && section.dataset.kind !== 'all');
      });
      tabs.forEach(tab => tab.classList.toggle('active', tab.dataset.filter === kind));
      search();
    }

    input.addEventListener('input', search);
    prev.addEventListener('click', () => move(current - 1));
    next.addEventListener('click', () => move(current + 1));
    tabs.forEach(tab => tab.addEventListener('click', () => filter(tab.dataset.filter)));
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(HTML, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
