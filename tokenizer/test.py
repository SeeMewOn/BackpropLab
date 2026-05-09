"""
Train our Tokenizers on some data, just to see them in action.
The whole thing runs in ~25 seconds on my laptop.
"""

import os
import time

from tokenizer.tokenizers.basic_tokenizer import BasicTokenizer
from tokenizer.tokenizers.regex_tokenizer import RegexTokenizer

# open some text and train a vocab of 512 tokens
text1 = open("texts/text1.txt", "r", encoding="utf-8").read()
text2 = open("texts/text2.txt", "r", encoding="utf-8").read()
text3 = open("texts/text3.txt", "r", encoding="utf-8").read()
# text4 = open("texts/text4.txt", "r", encoding="utf-8").read()
text = text1 + text2 + text3

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([BasicTokenizer, RegexTokenizer], ["basic", "regex"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 1024, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")