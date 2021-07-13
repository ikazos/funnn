import argparse
import fileinput

from transformers import pipeline

unmasker = pipeline("fill-mask", model="distilbert-base-uncased")

def unmask_sentences(sentences):
    return unmasker(sentences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fill in masked tokens in input sentences."
    )

    parser.add_argument(
        "inputs",
        metavar="INPUTS",
        type=str,
        nargs="*",
        help="paths to input files (if omitted, script reads from standard input)"
    )

    args = parser.parse_args()
    inputs = args.inputs

    sentences = []
    for line in fileinput.input(files=inputs):
        sentences.append(line)

    result = unmask_sentences(sentences)
    print(result)