import json
import sys
from pathlib import Path


def validate_dataset_a(file_path):
    errors = []
    warnings = []

    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    style_counts = {}

    for i, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: JSON parse failed - {e}")
            continue

        required_fields = ['instruction', 'input', 'output']
        for field in required_fields:
            if field not in record:
                errors.append(f"Line {i}: missing required field '{field}'")

        input_fields = ['topic', 'key_points', 'tone_preset']
        for field in input_fields:
            if field not in record.get('input', {}):
                errors.append(f"Line {i}: input missing required field '{field}'")

        valid_tones = ['casual_sarcastic', 'professional', 'marketing', 'academic', 'storytelling']
        tone = record.get('input', {}).get('tone_preset', '')
        if tone not in valid_tones:
            errors.append(f"Line {i}: invalid tone_preset '{tone}'")
        else:
            style_counts[tone] = style_counts.get(tone, 0) + 1

        if len(record.get('output', '')) < 100:
            warnings.append(f"Line {i}: output too short ({len(record.get('output', ''))} chars)")
        if len(record.get('output', '')) > 1000:
            warnings.append(f"Line {i}: output too long ({len(record.get('output', ''))} chars)")

    print("\n=== Dataset A Validation Report ===")
    print(f"Total records: {len(lines)}")
    print("\nStyle distribution:")
    for tone, count in sorted(style_counts.items()):
        print(f"  {tone}: {count}")

    if errors:
        print(f"\nFail: {len(errors)} error(s):")
        for err in errors[:10]:
            print(f"  - {err}")

    if warnings:
        print(f"\nWarn: {len(warnings)} warning(s):")
        for warn in warnings[:10]:
            print(f"  - {warn}")

    return len(errors) == 0


def validate_dataset_b(file_path):
    errors = []
    warnings = []

    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

    style_counts = {}
    score_distribution = {'high': 0, 'medium': 0, 'low': 0}

    for i, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: JSON parse failed - {e}")
            continue

        required_fields = ['text', 'tone_preset', 'style_score', 'style_analysis']
        for field in required_fields:
            if field not in record:
                errors.append(f"Line {i}: missing required field '{field}'")

        valid_tones = ['casual_sarcastic', 'professional', 'marketing', 'academic', 'storytelling']
        tone = record.get('tone_preset', '')
        if tone not in valid_tones:
            errors.append(f"Line {i}: invalid tone_preset '{tone}'")
        else:
            style_counts[tone] = style_counts.get(tone, 0) + 1

        score = record.get('style_score', 0)
        if not (0 <= score <= 100):
            errors.append(f"Line {i}: style_score {score} out of range [0, 100]")
        else:
            if score >= 80:
                score_distribution['high'] += 1
            elif score >= 60:
                score_distribution['medium'] += 1
            else:
                score_distribution['low'] += 1

        analysis_fields = ['perplexity', 'length_variance', 'vocabulary_match']
        for field in analysis_fields:
            if field not in record.get('style_analysis', {}):
                errors.append(f"Line {i}: style_analysis missing field '{field}'")

    print("\n=== Dataset B Validation Report ===")
    print(f"Total records: {len(lines)}")
    print("\nStyle distribution:")
    for tone, count in sorted(style_counts.items()):
        print(f"  {tone}: {count}")
    print("\nScore distribution:")
    print(f"  high (80-100):  {score_distribution['high']}")
    print(f"  medium (60-79): {score_distribution['medium']}")
    print(f"  low (<60):      {score_distribution['low']}")

    if errors:
        print(f"\nFail: {len(errors)} error(s):")
        for err in errors[:10]:
            print(f"  - {err}")

    if warnings:
        print(f"\nWarn: {len(warnings)} warning(s):")
        for warn in warnings[:10]:
            print(f"  - {warn}")

    return len(errors) == 0


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python validate_data.py <dataset_a|dataset_b> <file_path>")
        sys.exit(1)

    dataset_type = sys.argv[1]
    file_path = sys.argv[2]

    if dataset_type == 'dataset_a':
        success = validate_dataset_a(file_path)
    elif dataset_type == 'dataset_b':
        success = validate_dataset_b(file_path)
    else:
        print(f"Unknown dataset type: {dataset_type}")
        sys.exit(1)

    sys.exit(0 if success else 1)
