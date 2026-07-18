import json
import sys
import argparse


def convert_b_to_alpaca(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    converted = []
    for line in lines:
        record = json.loads(line)
        text = record['text']
        tone_preset = record['tone_preset']
        style_score = record['style_score']
        analysis = record['style_analysis']

        alpaca_record = {
            "instruction": "请分析以下文本的风格特征，并给出风格一致性评分。",
            "input": text,
            "output": json.dumps({
                "tone_preset": tone_preset,
                "style_score": style_score,
                "perplexity": analysis["perplexity"],
                "length_variance": analysis["length_variance"],
                "vocabulary_match": analysis["vocabulary_match"]
            }, ensure_ascii=False)
        }
        converted.append(alpaca_record)

    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Converted {len(converted)} records")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Dataset B regression format to Alpaca format')
    parser.add_argument('--input', required=True, help='Input JSONL file (regression format)')
    parser.add_argument('--output', required=True, help='Output JSONL file (Alpaca format)')
    args = parser.parse_args()
    convert_b_to_alpaca(args.input, args.output)
