"""Word surprisal of a target given a context, for any causal LM (todo §4).

    python score_lm.py --model DIR --name NAME --requests IN.tsv --out OUT.tsv

IN.tsv has columns req_id, context, target (from lm_items.py), both plain
words separated by spaces.  OUT.tsv has one row per request:

    req_id, model, n_words, surprisal (bits, whitespace-corrected, summed over
    the target's words), surprisal_raw (bits, uncorrected), words (each target
    word's corrected surprisal, comma separated)

The input is a BOS token followed by every word with a leading space, which
is how GPT-2 and Pythia's tokenisers mark the start of a word.  A word's raw
surprisal is the sum over its subword tokens.  The correction of Oh and
Schuler (2024) moves the probability that the word ends there into the
word: with B(t) the probability that the token after position t starts a
new word (a token with a leading space, a newline, or end of text),

    log P(word) = sum of its tokens' log probs + log B(last) - log B(before)

Runs on GPU if there is one, else on CPU; a 160M model needs minutes for
some tens of thousands of short requests.
"""

import argparse
import csv
import hashlib
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TSV = dict(delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
LN2 = math.log(2)


def word_start_ids(tok):
    """Token ids that begin a new word: leading space, newline, end of text."""
    ids = [i for t, i in tok.get_vocab().items()
           if t.startswith(('Ġ', 'Ċ'))]          # 'Ġ', 'Ċ'
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    return torch.tensor(sorted(set(ids)))


def encode(tok, words):
    """Token ids per word, each word with its leading space."""
    return [tok.encode(' ' + w, add_special_tokens=False) for w in words]


@torch.no_grad()
def score_batch(model, batch, bos, starts, device):
    """batch: list of (context word ids, target word ids). Returns per item a
    list of (raw log prob, corrected log prob) per target word, in nats."""
    seqs, spans = [], []
    for ctx, tgt in batch:
        ids = [bos] + [t for w in ctx for t in w]
        span = []
        for w in tgt:
            span.append((len(ids), len(ids) + len(w)))     # [a, b) positions
            ids += w
        seqs.append(ids)
        spans.append(span)
    width = max(map(len, seqs))
    input_ids = torch.full((len(seqs), width), bos, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for n, ids in enumerate(seqs):
        input_ids[n, :len(ids)] = torch.tensor(ids)
        mask[n, :len(ids)] = 1
    logits = model(input_ids=input_ids.to(device),
                   attention_mask=mask.to(device)).logits.float()
    logp = torch.log_softmax(logits, dim=-1).cpu()
    word_start = torch.logsumexp(logp[:, :, starts], dim=-1)   # log B(t)
    out = []
    for n, (ids, span) in enumerate(zip(seqs, spans)):
        words = []
        for a, b in span:
            raw = sum(logp[n, p - 1, ids[p]].item() for p in range(a, b))
            corr = raw + word_start[n, b - 1].item() - word_start[n, a - 1].item()
            words.append((raw, corr))
        out.append(words)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='local model directory')
    ap.add_argument('--name', required=True, help='label written to --out')
    ap.add_argument('--requests', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--batch', type=int, default=32)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32)
    model.to(device).eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    starts = word_start_ids(tok)
    # GPT-2 and Pythia mark a word start with 'Ġ'; a tokeniser that does not
    # would make the correction silently wrong
    assert len(starts) > 10000, f'only {len(starts)} word-start tokens'
    print(f'{args.name}: {args.model} on {device}, torch {torch.__version__}, '
          f'bos {bos}, {len(starts)} word-start tokens', flush=True)

    with open(args.requests, newline='') as f:
        reqs = list(csv.DictReader(f, **TSV))
    print(f'{len(reqs)} requests from {args.requests} '
          f'(md5 {hashlib.md5(args.requests.read_bytes()).hexdigest()})', flush=True)

    # longest first, so a batch holds sequences of similar length
    enc = [(r, encode(tok, r['context'].split()), encode(tok, r['target'].split()))
           for r in reqs]
    order = sorted(range(len(enc)), key=lambda n: -sum(
        map(len, enc[n][1] + enc[n][2])))
    results = {}
    t0 = time.time()
    for start in range(0, len(order), args.batch):
        idx = order[start:start + args.batch]
        scored = score_batch(model, [(enc[n][1], enc[n][2]) for n in idx],
                             bos, starts, device)
        for n, words in zip(idx, scored):
            results[n] = words
        if start // args.batch % 50 == 0:
            print(f'  {start + len(idx)}/{len(order)}  {time.time() - t0:.0f} s',
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n', **TSV)
        w.writerow(['req_id', 'model', 'n_words', 'surprisal', 'surprisal_raw',
                    'words'])
        for n, (r, _, _) in enumerate(enc):
            words = results[n]
            w.writerow([r['req_id'], args.name, len(words),
                        f'{-sum(c for _, c in words) / LN2:.4f}',
                        f'{-sum(x for x, _ in words) / LN2:.4f}',
                        ','.join(f'{-c / LN2:.4f}' for _, c in words)])
    print(f'wrote {args.out} in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    sys.exit(main())
