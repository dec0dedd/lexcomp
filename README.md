# LexComp: Python library for text complexity assessment

LexComp is a straightforward Python library designed to measure text complexity through various statistics. Currently in its development phase, its functionalities are limited, but future plans include enhancing metrics with transformer models (such as BERT) and incorporating additional metrics.

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
First text's complexity: 0.03551881
Second text's complexity: 0.5091553
```

## Data sources
1. CommonLit corpus - [source](https://github.com/scrosseye/CLEAR-Corpus)
2. OneStopEnglish corpus - [source](https://github.com/nishkalavallabhi/OneStopEnglishCorpus)
3. CEFR-based Short Answer corpus - [source](https://cental.uclouvain.be/team/atack/cefr-asag/)