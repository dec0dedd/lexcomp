# LexComp: Python library for text complexity analysis

LexComp is a quick and easy Python library designed to measure text complexity through various statistics and ML models. Models and metrics are still work in progress so more changes might be introduced in the future.

## How does it work?

LexComp evaluates text complexity by blending DistilBERT embeddings with classic readability metrics and combining them through XGBoost decision trees. DistilBERT helps capture the subtleties of language, while metrics like Flesch-Kincaid add insights on sentence length and word difficulty. Together, they create a well-rounded complexity score that offers a more complete picture of how readable a piece of text really is.

## Example usage

```py
from lexcomp import Model

text1 = """
Amidst the complexities of modern society, technology emerges as a transformative force, reshaping how we acquire knowledge and communicate. Its pervasive influence fosters an environment ripe for innovation, allowing individuals to transcend geographical barriers and engage in collaborative endeavors. As we navigate this digital landscape, the imperative to harness these advancements responsibly becomes increasingly paramount, ensuring that the pursuit of progress aligns with ethical considerations and societal well-being.
"""

text2 = "Technology helps us learn and talk. It lets friends work together, even if they are far away. We should use technology nicely so it helps everyone."

mdl = Model()
print(f"First text's complexity: {mdl.predict_text(text1)[0]}")
print(f"Second text's complexity: {mdl.predict_text(text2)[0]}")
```

The output should be
```
First text's complexity: 0.7669976353645325
Second text's complexity: 0.129152312874794
```

## Data sources
1. CommonLit corpus - [source](https://github.com/scrosseye/CLEAR-Corpus)
2. OneStopEnglish corpus - [source](https://github.com/nishkalavallabhi/OneStopEnglishCorpus)
3. CEFR-based Short Answer corpus - [source](https://cental.uclouvain.be/team/atack/cefr-asag/)