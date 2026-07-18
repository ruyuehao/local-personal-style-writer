"""
Qoder AI Skill Inference
=======================
使用 OpenVINO INT4 模型进行推理：
  - personal: Qwen3-8B 个人风格生成
  - analysis: Qwen2.5-0.5B 风格分析

用法:
  python qoder_inference.py personal --model-dir models/qwen3_personal_int4 --input "我的主题" --key-points 要点1 要点2
  python qoder_inference.py analysis --model-dir models/qwen2.5_int4 --input "待分析文本"
  python qoder_inference.py personal --model-dir ... --input "主题" --json   # JSON 输出供程序调用
"""

import json
import re
import sys
import argparse
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

from device_manager import device_manager


def _strip_thinking(text: str) -> str:
    """移除 Qwen3 输出中的 <think>...</think> 思考块。"""
    return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()


def load_model(model_dir: str, model_type: str = "personal", **kwargs):
    device = device_manager.pick(model_type)
    model = OVModelForCausalLM.from_pretrained(model_dir, device=device, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer


def generate_personal_style(
    model,
    tokenizer,
    topic: str,
    key_points: list[str] | None = None,
    target_length: str | None = None,
    tone_preset: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    prompt_override: str | None = None,
) -> str:
    if prompt_override:
        prompt = prompt_override
    else:
        query_parts = [f"主题：{topic}"]
        if key_points:
            query_parts.append(f"要点：{'；'.join(key_points)}")
        if target_length:
            query_parts.append(f"目标长度：{target_length}")
        if tone_preset:
            query_parts.append(f"风格：{tone_preset}")

        query = "\n".join(query_parts)
        # 插入空 <think></think> 禁用 Qwen3 思考模式（与训练模板 qwen3_nothink 一致）
        prompt = f"<|im_start|>user\n请根据以下要求写一篇文章。\n{query}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = result[result.find("assistant\n") + len("assistant\n"):] if "assistant\n" in result else result
    response = _strip_thinking(response)
    return response.strip()


# 字段别名：英文键 + 中文同义词，使基座模型用中文输出时也能被抽取
_FIELD_ALIASES = {
    "style_score": ["style_score", "风格一致性评分", "风格评分", "评分", "score"],
    "perplexity": ["perplexity", "困惑度", "困惑"],
    "length_variance": ["length_variance", "句长方差", "长度方差", "句式方差"],
    "vocabulary_match": ["vocabulary_match", "词汇匹配", "词汇契合", "词汇契合度"],
}


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _normalize_field(field: str, raw) -> float | None:
    """把任意来源的数值归一到训练契约区间。

    - style_score: 0-100（兼容模型偶尔输出 0-1 比例）
    - perplexity: 10-200（训练量纲，直接透传，绝不压到 0-1）
    - length_variance: 0-100（训练量纲，直接透传，绝不压到 0-1）
    - vocabulary_match: 0-1
    """
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return None
    if field == "style_score":
        return max(0.0, min(100.0, val * 100.0 if val <= 1.0 else val))
    if field in ("perplexity", "length_variance"):
        return val
    if field == "vocabulary_match":
        return _clamp01(val)
    return val


def _extract_field(text: str, field: str) -> float | None:
    """从模型回复中抽取某字段数值，兼容英文 JSON 键与中文同义词（含「评分90」「达到90分」）。"""
    for alias in _FIELD_ALIASES.get(field, [field]):
        patterns = [
            rf'"{re.escape(alias)}"\s*:\s*([\d.]+)',
            rf"'{re.escape(alias)}'\s*:\s*([\d.]+)",
            rf"{re.escape(alias)}\s*[：:]\s*([\d.]+)",
            rf"{re.escape(alias)}\s*([\d.]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return _normalize_field(field, m.group(1))
    # style_score 额外兜底：基座模型常写「评分90」「达到90分」而无英文键
    if field == "style_score":
        for pat in [r"评分[为是：: ]*\s*([\d.]+)", r"达到\s*([\d.]+)\s*分", r"(\d+(?:\.\d+)?)\s*分"]:
            m = re.search(pat, text)
            if m:
                return _normalize_field(field, m.group(1))
    return None


def _deterministic_analysis(text: str) -> dict:
    """模型未吐出结构化数值时的确定性兜底：基于文本统计给出可复现的指标，
    且数值已映射到训练契约量纲（length_variance 0-100 / vocabulary_match 0-1 / perplexity 10-200）。"""
    sents = [s for s in re.split(r"[。！？!?；;\n]", text) if s.strip()]
    if not sents:
        sents = [text]
    lens = [len(s) for s in sents]
    mean = sum(lens) / len(lens)
    var = sum((x - mean) ** 2 for x in lens) / len(lens)
    length_variance = min(100.0, max(0.0, var / 2.0))          # 句长方差，封顶 100
    words = re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+", text)
    ttr = (len(set(words)) / len(words)) if words else 0.0
    vocabulary_match = _clamp01(ttr * 2.5)
    perplexity = min(200.0, max(10.0, mean * 2.0))              # 以平均句长近似，限到 10-200
    style_score = round(max(0.0, min(100.0,
        100.0 * (1.0 - 0.5 * (length_variance / 100.0)) * (0.4 + 0.6 * vocabulary_match))))
    return {
        "style_score": float(style_score),
        "perplexity": round(perplexity, 3),
        "length_variance": round(length_variance, 3),
        "vocabulary_match": round(vocabulary_match, 3),
    }


def _extract_numeric_field(text: str, field_name: str) -> float | None:
    patterns = [
        rf'"{field_name}"\s*:\s*([\d.]+)',
        rf"'{field_name}'\s*:\s*([\d.]+)",
        rf'{field_name}[：:]\s*([\d.]+)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


def _extract_json(text: str):
    """从模型回复中提取 JSON 对象，兼容纯 JSON / ```json 代码围栏 / 前后夹带杂文本。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    return None


def analyze_style(
    model,
    tokenizer,
    text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    user_style: str | None = None,   # 保留参数兼容调用方；训练数据不含 user_style，故不注入
) -> dict:
    # 与微调训练模板严格对齐（convert_b_to_alpaca.py + LLaMA-Factory qwen 模板）：
    #   system: "You are a helpful assistant."
    #   user:   "请分析以下文本的风格特征，并给出风格一致性评分。\n\n" + text
    # 模型被训练为直接输出 JSON：{tone_preset, style_score, perplexity, length_variance, vocabulary_match}
    instruction = "请分析以下文本的风格特征，并给出风格一致性评分。"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{instruction}\n\n{text}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    # 只取模型新生成的部分（不含 prompt），避免取到输入回声
    generated = outputs[0][input_len:]
    response = tokenizer.decode(generated, skip_special_tokens=True)
    response = _strip_thinking(response).strip()

    # 1) 解析模型输出的 JSON（训练目标格式）
    parsed = _extract_json(response)
    fields: dict = {}
    if isinstance(parsed, dict):
        tp = parsed.get("tone_preset")
        if tp:
            fields["tone_preset"] = str(tp)
        for k in ("style_score", "perplexity", "length_variance", "vocabulary_match"):
            raw = parsed.get(k)
            if raw is None:
                for alias in _FIELD_ALIASES.get(k, [k]):
                    if alias in parsed:
                        raw = parsed[alias]
                        break
            if raw is not None:
                fields[k] = raw

    # 2) 数值字段从回复文本里兜底抽取（英文键 + 中文同义词）
    for k in ("style_score", "perplexity", "length_variance", "vocabulary_match"):
        if fields.get(k) is None:
            v = _extract_field(response, k)
            if v is not None:
                fields[k] = v

    # 3) 仍缺失的字段用确定性兜底（保证永不全 null），兜底值已映射到训练量纲
    fallback = _deterministic_analysis(text)
    for k in ("style_score", "perplexity", "length_variance", "vocabulary_match"):
        if fields.get(k) is None:
            fields[k] = fallback[k]

    # 4) 归一量纲（perplexity / length_variance 透传，不压到 0-1）
    norm = {
        "style_score": _normalize_field("style_score", fields["style_score"]),
        "perplexity": _normalize_field("perplexity", fields["perplexity"]),
        "length_variance": _normalize_field("length_variance", fields["length_variance"]),
        "vocabulary_match": _normalize_field("vocabulary_match", fields["vocabulary_match"]),
    }

    # 模型被训练为只吐 JSON（非散文），故 description 由结构化字段规则生成，不取原始回复
    description = _rule_based_description(norm)
    return {
        "tone_preset": fields.get("tone_preset"),
        "style_score": round(norm["style_score"]),
        "style_analysis": {
            "perplexity": round(norm["perplexity"], 3),
            "length_variance": round(norm["length_variance"], 3),
            "vocabulary_match": round(norm["vocabulary_match"], 3),
        },
        "description": description,
    }


def _clean_description(text: str) -> str:
    """从模型原始回复中提取自然语言风格描述，去掉 JSON 碎片与尾部评分噪声。"""
    text = re.sub(r"\{[^{}]*\}", "", text)            # 去 JSON 块
    text = re.sub(r"style_score\s*[:：]?\s*\d+", "", text, flags=re.I)
    text = re.sub(r"达到\s*\d+\s*分", "", text)
    text = text.replace("assistant", "").replace("user", "")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip("，。、 ")
    if text and text[-1] not in "。！？.!?":
        text += "。"
    return text[:400]


def _rule_based_description(fields: dict) -> str:
    """模型未吐出描述文本时的规则兜底：基于数值拼出可读结论。"""
    lv = fields.get("length_variance") or 0.0
    vm = fields.get("vocabulary_match") or 0.0
    ppl = fields.get("perplexity") or 0.0
    sent = f"风格一致性评分 {round(fields.get('style_score') or 0)} 分。"
    sent += "句式" + ("长短错落、富于变化" if lv > 0.5 else "规整统一") + "，"
    sent += "词汇" + ("与个人口语化风格高度契合" if vm > 0.6 else "偏书面") + "，"
    sent += "语言" + ("通顺流畅" if ppl < 0.5 else "略有阻滞") + "。"
    return sent


def _build_description(response: str, fields: dict) -> str:
    cleaned = _clean_description(response)
    if cleaned:
        return cleaned
    return _rule_based_description(fields)


def main():
    parser = argparse.ArgumentParser(description="Qoder AI Skill Inference")
    parser.add_argument("model", nargs="?", choices=["personal", "analysis"], help="Model to use")
    parser.add_argument("--model-dir", required=False, help="Path to INT4 model directory")
    parser.add_argument("--input", help="Input text")
    parser.add_argument("--key-points", nargs="*", help="Key points (personal only)")
    parser.add_argument("--target-length", help="Target length (personal only)")
    parser.add_argument("--tone-preset", help="Tone preset (personal only)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--device-info", action="store_true", help="Show device allocation info")

    args = parser.parse_args()

    if args.device_info:
        print(json.dumps(device_manager.summary, ensure_ascii=False, indent=2))
        return

    print(f"Loading model from {args.model_dir}...", file=sys.stderr)
    model, tokenizer = load_model(args.model_dir, model_type=args.model, use_cache=False)

    if args.model == "personal":
        result = generate_personal_style(
            model, tokenizer,
            topic=args.input,
            key_points=args.key_points,
            target_length=args.target_length,
            tone_preset=args.tone_preset,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        result = analyze_style(
            model, tokenizer,
            text=args.input,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result)


if __name__ == "__main__":
    main()