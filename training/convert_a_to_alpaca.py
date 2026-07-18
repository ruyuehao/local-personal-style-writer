import json
import sys
import argparse


def convert_a_to_alpaca(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    converted = []
    for line in lines:
        record = json.loads(line)
        inp = record['input']

        pieces = [f"主题：{inp['topic']}"]
        if inp.get('key_points'):
            pieces.append(f"要点：{'；'.join(inp['key_points'])}")
        if inp.get('target_length'):
            pieces.append(f"目标长度：{inp['target_length']}")
        if inp.get('tone_preset'):
            pieces.append(f"风格：{inp['tone_preset']}")
        if inp.get('preserve_terms'):
            pieces.append(f"保留术语：{'，'.join(inp['preserve_terms'])}")

        query_str = "\n".join(pieces)

        alpaca_record = {
            "instruction": record['instruction'],
            "input": query_str,
            "output": record['output']
        }
        converted.append(alpaca_record)

    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in converted:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"Converted {len(converted)} records")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Dataset A object-input to Alpaca string format')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output JSONL file (Alpaca string format)')
    args = parser.parse_args()
    convert_a_to_alpaca(args.input, args.output)
