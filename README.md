# Knowledge Graph Construction with LLMs

Refernce - [Anthropic claude-cookbooks knowledge graph guide](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/knowledge_graph/guide.ipynb) 

---

## The Core Idea

There are two ways to build a knowledge graph depending on your data:

**Unstructured data** (Wikipedia articles, fault tickets, maintenance reports, engineering PDFs)
The data has no rows or columns, it is prose. An LLM reads each document and extracts entities and relations into a typed schema. No labeled training data required. No domain-specific NLP model required.

**Structured data** (Spreadsheets,  XML exports, database tables)
The schema is already known. A deterministic parser reads the rows and columns directly into the graph. No LLM needed for extraction.

This notebook demonstrates the first path. The querying and summarisation cells at the end work identically regardless of how the graph was built.

---

<img width="1440" height="1398" alt="image" src="https://github.com/user-attachments/assets/309b63e1-19e1-44bb-b75d-c4790b9da6b6" />


---

## Why LLM for Extraction

Classical named entity recognition and relation extraction require labeled training data per domain. A separate coreference model is needed to resolve pronouns. A separate alias resolution model is needed for cross-name variants. Each requires domain-specific training.

The LLM collapses all into a single prompt call. No training data. No domain-specific model. The structured output response always validates against the Pydantic schema, no regex, no JSON decode errors.

The critical limitation: the LLM can miss entities, hallucinate relations, or over-merge aliases. This is why the evaluation cell exists, to measure extraction quality against a gold standard and guide prompt iteration.

---

## Why String Similarity Fails for Entity Resolution

Edit distance catches typos: `"Neil Armstong"` → `"Neil Armstrong"`.

It cannot catch `"Edwin Aldrin"` → `"Buzz Aldrin"`. Zero character overlap. Edwin is his birth name; Buzz is his legal name since 1988. The LLM reads the descriptions like *Lunar Module Pilot on Apollo 11, second person to walk on the Moon*, and reasons that both names refer to the same person.

This is the core motivation for using an LLM in the resolution step rather than a classical algorithm.


## Adapting to a New Domain

Three things to change:

**1. Redefine the entity type taxonomy.**
```python
# Automotive fault diagnosis
EntityType = Literal["Sensor", "SIGNAL", "BUS", "FUNCTION"]
```

**2. Redesign EntityProfile for your domain.**
```python
# fault tickets
class EntityProfile(BaseModel):
    summary: str
    affected_units: list[str]
    fault_pattern: Literal["timeout", "value_out_of_range", "missing", "other"]
    severity: Literal["critical", "major", "minor"]
    first_seen: str
```

**3. Adjust the extraction prompt** to define what counts as a central entity for your document type.

`Entity` and `Relation` are generic enough to reuse unchanged across domains.

---

## Structured Data Path

For structured sources like a Matrix or XML export, we parse directly into the same NetworkX node/edge schema. Then use below those cells for visualisation, summarisation, and querying.

The key insight: the graph schema is domain-independent. How you populate it, LLM extraction or deterministic parser depends only on whether your source data is unstructured or structured.

---

## Setup

```bash
pip install openai networkx matplotlib python-dotenv pydantic httpx
```

Create a `.env` file:
```
Define your model realted varibles
```

Run `knowledge_graph_construction.ipynb` top to bottom.

---
## How it beats RAG

RAG retrieves chunks and hopes the answer lives in one chunk.
Knowledge graphs don't retrieve, they traverse.

### The retrieval problem

```
Chunk A:  "Armstrong commanded Apollo 11"
Chunk B:  "Apollo 11 launched from Kennedy Space Center"

RAG question: "Where did Armstrong's mission launch from?"
→ If A and B never co-occur in one retrieved chunk, RAG cannot connect them.
→ The graph traverses the edge chain directly.
```
We could do Hierarchical RAG, but if two chuck are in complete differnt documents this is the efficient approach.
Every edge is an explicit reasoning step. Multi-hop questions that require
connecting facts across documents are answered by following edges, not by
hoping two facts happened to land in the same retrieval window.

### Grounded vs ungrounded

`ask()` is called twice on the same question:

```
WITHOUT context  →  model answers from pretraining
WITH context     →  model must cite specific edges from your graph
```

For public knowledge like Apollo, the ungrounded answer is often richer 
the model already knows the facts. For private data like  fault tickets,
 diagnostic exports, Matrix signal flows the ungrounded answer is
useless. The model has never seen your fleet data.

```
Public data   →  both paths work, ungrounded is faster
Private data  →  only grounded path works, full stop
```

---

## Evaluation

A knowledge graph pipeline has two failure points that need to be measured
separately: extraction quality and resolution quality.

### Metrics

```
Precision  =  correct extractions / total extractions
Recall     =  correct extractions / total gold entities
F1         =  2 * P * R / (P + R)
```

**High precision, low recall** LLM too conservative. Relax the
"central entities only" instruction in the extraction prompt.

**High recall, low precision** LLM over-extracting. Tighten the
selectivity instruction.

## References

- Original cookbook: https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/knowledge_graph/guide.ipynb
- Azure OpenAI structured outputs: https://learn.microsoft.com/azure/ai-services/openai/how-to/structured-outputs
- NetworkX documentation: https://networkx.org/documentation/stable/
