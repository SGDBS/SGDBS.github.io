"""
批量为 jokes.json 中的每条苏联笑话生成笑点解析。
运行前请确保已安装 anthropic：pip install anthropic
API Key 通过环境变量 ANTHROPIC_API_KEY 传入，或直接在下方填写。

用法：
    python generate_analyses.py

支持断点续跑：已有 analysis 的条目会跳过，直接保留。
"""

import json
import time
import os
import anthropic

# ── 配置 ──────────────────────────────────────────────────────────────
JOKES_PATH = os.path.join(os.path.dirname(__file__), "source", "js", "jokes.json")
MODEL = "claude-haiku-4-5-20251001"   # 用 Haiku 节省成本
MAX_TOKENS = 220
SLEEP_EVERY = 10   # 每处理 N 条暂停一次，避免触发速率限制
SLEEP_SEC = 1.0
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一位熟悉苏联历史与政治的幽默评论员。"
    "请用简洁、风趣的中文（1-3句话）解析笑话的笑点，"
    "说明其中涉及的历史背景、政治讽刺或语言技巧，"
    "最后必须以"令人忍俊不禁"结尾。"
    "不要复述笑话原文，直接点题。"
)

def build_prompt(joke_text: str) -> str:
    return f"请解析以下苏联政治笑话的笑点：\n\n{joke_text}"


def generate_analysis(client: anthropic.Anthropic, text: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(text)}],
    )
    return msg.content[0].text.strip()


def main():
    client = anthropic.Anthropic()  # 读取 ANTHROPIC_API_KEY 环境变量

    with open(JOKES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    updated = 0

    for i, item in enumerate(data):
        # 兼容旧格式（纯字符串）和新格式（对象）
        if isinstance(item, str):
            data[i] = {"text": item, "analysis": ""}
            item = data[i]

        # 已有解析则跳过（支持断点续跑）
        if item.get("analysis"):
            continue

        text = item["text"]
        print(f"[{i+1}/{total}] 生成解析……")

        try:
            analysis = generate_analysis(client, text)
            # 确保结尾为"令人忍俊不禁"
            if not analysis.endswith("令人忍俊不禁"):
                # 若模型没带上结尾，强制拼接
                analysis = analysis.rstrip("。！？…") + "，令人忍俊不禁。"
            item["analysis"] = analysis
            updated += 1
        except Exception as e:
            print(f"  ⚠ 第 {i+1} 条失败：{e}，已跳过，下次可继续。")
            item["analysis"] = ""

        # 每 N 条保存一次（防止中途崩溃丢数据）
        if (i + 1) % SLEEP_EVERY == 0:
            with open(JOKES_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 已保存进度（{i+1}/{total}）")
            time.sleep(SLEEP_SEC)

    # 最终保存
    with open(JOKES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n完成！本次新生成 {updated} 条解析，共 {total} 条。")


if __name__ == "__main__":
    main()
