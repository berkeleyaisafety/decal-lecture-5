"""Chat with the model while steering it.

Usage:
    python chat.py vectors/g3-syco-v1.pt --alpha 6             # layer from settings.py
    python chat.py vectors/g3-syco-v1.pt --layer 12 --alpha -4
    python chat.py base                                       # no steering

Commands during the chat:
    /alpha 8     change the steering strength (0 turns it off)
    /layer 12    change the layer
    /reset       forget the conversation so far
    /quit        exit
"""

import argparse

from interp import generate, load, load_vector, steering, steering_defaults


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("vector", help="vectors/<name>.pt from extract.py, or 'base' for no steering")
    parser.add_argument("--layer", type=int, default=None, help="default: settings.py for this trait")
    parser.add_argument("--alpha", type=float, default=None, help="default: settings.py for this trait")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    vec = None if args.vector == "base" else load_vector(args.vector)
    default_layer, default_alpha = steering_defaults(vec) if vec else (16, 0.0)
    layer = args.layer or default_layer
    alpha = (args.alpha if args.alpha is not None else default_alpha) if vec else 0.0
    load()
    history = []
    print("Type a message, or /alpha N, /layer N, /reset, /quit.")

    while True:
        tag = f"layer {layer}, alpha {alpha:+g}" if vec else "base"
        try:
            text = input(f"\n[{tag}] you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            history = []
            print("(conversation cleared)")
            continue
        if text.startswith("/alpha ") or text.startswith("/layer "):
            if vec is None:
                print("(no vector loaded: run with a vectors/<name>.pt file to steer)")
                continue
            command, value = text.split(maxsplit=1)
            if command == "/alpha":
                alpha = float(value)
            else:
                layer = int(value)
            continue

        history.append({"role": "user", "content": text})
        if vec is not None and alpha != 0:
            with steering(layer, vec["direction"][layer - 1], [alpha]):
                reply = generate(history, max_new_tokens=args.max_tokens, temperature=args.temperature)[0]
        else:
            reply = generate(history, max_new_tokens=args.max_tokens, temperature=args.temperature)[0]
        history.append({"role": "assistant", "content": reply})
        print(f"\nmodel> {reply}")


if __name__ == "__main__":
    main()
