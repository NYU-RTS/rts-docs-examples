
### How to run the script:
```
ss19980@ITS-JQKQGQQMTX ~/D/p/r/g/rag (main)> uv run rag_basic.py --help
usage: rag_basic [-h] url query

Basic demonstration of Retrieval-Augmented Generation

positional arguments:
  url         URL of website to store in vector db after parsing and chunking
  query       query to LLM which can benefit from information at URL

options:
  -h, --help  show this help message and exit
ss19980@ITS-JQKQGQQMTX ~/D/p/r/g/rag (main)> 
```

### Sample response on local machine:

```
ss19980@ITS-JQKQGQQMTX ~/D/p/r/g/rag (main)> uv run rag_basic.py \
                                                 https://en.wikipedia.org/wiki/2024_Wimbledon_Championships \
                                                 "Who won the 2024 Wimbledon championships?"
/Users/ss19980/Documents/packages/rts-docs-examples/genai/rag/.venv/lib/python3.13/site-packages/milvus_lite/__init__.py:15: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import DistributionNotFound, get_distribution
Processing chunks: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 95/95 [00:27<00:00,  3.41it/s]
Query embedding vector (first 10 dims) is:  [-0.02074776403605938, 0.012319051660597324, 0.0028169124852865934, -0.07154089212417603, -0.02403954602777958, 0.020915305241942406, 0.007872640155255795, -0.0010293368250131607, -0.031030597165226936, 0.011616146191954613]
----------------------------------------------------------------

Retreived chunks and similarity scores:
("It was the 137th edition of the Wimbledon Championships and the third Grand Slam event of 2024. The gentlemen's singles title was won by defending champion Carlos Alcaraz, who defeated Novak Djokovic in a rematch of the previous year's final to lift his fourth Grand Slam title.[1] Barbora Krejčíková defeated Jasmine Paolini in the final to win the ladies' singles title.[2]", 0.8189371228218079)
('The 2024 Wimbledon Championships was a major tennis tournament that took place at the All England Lawn Tennis and Croquet Club in Wimbledon, London, England, comprising singles, doubles and mixed doubles play. Junior, wheelchair and Invitational tournaments were also scheduled.', 0.8104943037033081)
('Wimbledon 2024 gentlemen’s singles draw', 0.7985922694206238)
----------------------------------------------------------------

Generated response from LLM without additional context is:
The 2024 Wimbledon Championships haven't taken place yet! They are scheduled from July 1 to July 14, 2024.

Therefore, there's no winner to announce yet.
----------------------------------------------------------------

Generated response from LLM with additional context is:
Carlos Alcaraz won the gentlemen's singles title and Barbora Krejčíková won the ladies' singles title at the 2024 Wimbledon Championships.
----------------------------------------------------------------

E20250611 13:10:10.121642 262942 server.cpp:47] [SERVER][BlockLock][] Process exit
ss19980@ITS-JQKQGQQMTX ~/D/p/r/g/rag (main)>
```
