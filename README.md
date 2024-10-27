# LexComp: Python library for text complexity assessment

LexComp is a straightforward Python library designed to measure text complexity through various statistics. Currently in its development phase, its functionalities are limited, but future plans include enhancing metrics with transformer models (such as BERT) and incorporating additional metrics.

## Example usage

```py
from lexcomp import Model

text1 = """
everyday i get up at 8 a clock. I always turn my music on. To have a nice day. Firstly, i take my shower. In a second time I breakfast. Then i go to school. But that's depend about my timetable. I don't begin at 8h30 everday so sometimes I watch tv
"""

text2 = """
My answer to this question is yes, famous people do have a right to privacy. Artists, actors, musicians, politicians (and so on) don't need every single aspects of their life known and shown to the whole world. The only thing that should matter is if these people do their job right. As long as they don't break any rule, why should we know who they married and what their children look like or if they cheated,... Furthermore, media making money by invading privacy are often terrible. Examples of this are tabloids or reality show like the Kardashians. That is not good TV. It's getting harder and harder for celebrities to keep their private life private, with the increase of social media and everybody owning a camera phone. We should put ourselves in their shoes and ask us how we would like to be treated if we were in their place.
"""

mdl = Model()
print(f"First text's complexity: {mdl.predict_text(text1)[0]}")
print(f"Second text's complexity: {mdl.predict_text(text2)[0]}")
```

The output should be
```
First text's complexity: 0.03551881
Second text's complexity: 0.5091553
```