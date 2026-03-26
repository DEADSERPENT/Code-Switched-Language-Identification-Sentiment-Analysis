"""
Quick terminal demo for CoSwitchNLP — no web server needed.

Usage:
    python demo.py --model_dir ../models/coswitchnlp_v1
    python demo.py --model_dir ../models/coswitchnlp_v1 --text "yaar ye bohot achha tha"

Interactive mode (no --text flag) lets you type sentences one by one.
"""

import argparse
import sys

from inference import CoSwitchInference


LANG_COLORS = {
    "lang1": "\033[93m",   # yellow  — Hindi
    "lang2": "\033[94m",   # blue    — English
    "mixed": "\033[95m",   # magenta — Mixed
    "ne":    "\033[92m",   # green   — Named Entity
    "other": "\033[90m",   # gray
    "univ":  "\033[90m",   # gray
}
RESET = "\033[0m"
BOLD  = "\033[1m"

SENTIMENT_ICONS = {
    "positive": "✅ POSITIVE",
    "neutral":  "➖ NEUTRAL",
    "negative": "❌ NEGATIVE",
}


def display_result(text: str, result) -> None:
    print(f"\n{BOLD}Input:{RESET} {text}")
    print(f"{BOLD}{'─' * 60}{RESET}")

    # Token-level LID
    print(f"{BOLD}Token Language ID:{RESET}")
    parts = []
    for tp in result.tokens:
        color = LANG_COLORS.get(tp.language, "")
        label = tp.language.upper()[:2]
        parts.append(f"{color}{tp.token}{RESET}[{label}]")
    print("  " + "  ".join(parts))

    # Legend
    legend = "  Legend: "
    legend_items = [
        ("\033[93m", "HI=Hindi"),
        ("\033[94m", "EN=English"),
        ("\033[95m", "MX=Mixed"),
        ("\033[92m", "NE=NamedEntity"),
        ("\033[90m", "OT=Other"),
    ]
    print(legend + "  ".join(f"{c}{l}{RESET}" for c, l in legend_items))

    # Sentiment
    icon = SENTIMENT_ICONS.get(result.sentiment, result.sentiment.upper())
    print(f"\n{BOLD}Sentiment:{RESET} {icon}  ({result.sentiment_confidence*100:.1f}% confidence)")
    for label, score in sorted(result.sentiment_scores.items(), key=lambda x: -x[1]):
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {label:10s} {bar}  {score*100:.1f}%")

    # CMI
    cmi_pct = result.code_mixing_index * 100
    mixing_label = (
        "Low (mostly monolingual)"       if cmi_pct < 20 else
        "Moderate (some code-switching)" if cmi_pct < 50 else
        "High (heavy code-switching)"
    )
    print(f"\n{BOLD}Code-Mixing Index (CMI):{RESET} {cmi_pct:.0f}%  — {mixing_label}")

    # Language distribution
    print(f"{BOLD}Language distribution:{RESET}")
    for tag, frac in result.language_distribution.items():
        if frac > 0:
            print(f"  {tag:8s} {frac*100:.1f}%")

    print(f"{BOLD}{'─' * 60}{RESET}")


def main() -> None:
    p = argparse.ArgumentParser(description="CoSwitchNLP terminal demo")
    p.add_argument("--model_dir", default="../models/coswitchnlp_v1")
    p.add_argument("--text", default=None, help="Single sentence to analyse (optional)")
    args = p.parse_args()

    print(f"\n{BOLD}CoSwitchNLP — Hinglish Language ID & Sentiment Demo{RESET}")
    print("Loading deep learning model...\n")

    try:
        engine = CoSwitchInference(args.model_dir)
    except Exception as e:
        print(f"ERROR: Could not load model from '{args.model_dir}'")
        print(f"  {e}")
        print("\nMake sure you have run train.py first.")
        sys.exit(1)

    if args.text:
        result = engine.predict(args.text)
        display_result(args.text, result)
        return

    # Interactive loop
    print("Type Hinglish text and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        try:
            result = engine.predict(text)
            display_result(text, result)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
