# 📚 Evonarrator: Evolutionary Narrative-Driven Research Hypothesis Generation

An end-to-end academic paper analysis system that creates a complete closed loop from paper retrieval to research idea generation through the construction of citation knowledge graphs, deep information extraction, and relationship mining.

## 🌟 Core Features

This system realizes the full-process automated analysis of academic research through five core modules:

1. **Paper Retrieval & Citation Network Construction** - Six-step snowballing retrieval strategy
2. **Paper Download, Parsing & Deep Extraction** - Multi-agent collaborative extraction system
3. **Citation Relationship Enrichment** - Socket interface matching mechanism
4. **Knowledge Graph & Evolution Trajectory Mapping** - Deep evolutionary trajectory generation
5. **Research Idea Generation** - Creative combination based on defect pools and method libraries

---

## 🔍 Module 1: Paper Retrieval and Citation Network Construction

### Goal
- Retrieve papers highly relevant to the target topic.
- Build a rich and comprehensive citation relationship network.

### Method: Six-Step Snowballing Retrieval Pipeline

```text
Step 1: Foundational Seeds
├─ Call arXiv API to find classic papers in the field as seed nodes

Step 2: Forward Snowballing
├─ Who cited the Seed? → Find child nodes
└─ Retrieve cited relationships via OpenAlex

Step 3: Backward Snowballing
├─ Who did the Seed cite? → Find parent/ancestor nodes
└─ Trace technological origins

Step 4: Horizontal Supplement / Co-citation Mining
├─ Among child and parent nodes, who is frequently mentioned but missing in the library?
└─ Supplement missing key papers

🔄 Step 5 (Optional): Second-Round Snowballing Expansion
├─ Forward: Find child nodes from first-round papers
├─ Backward: Find parent nodes from first-round papers
└─ Co-citation: Analyze co-citation patterns of second-round papers

Step 6: Supplement Latest SOTA (State-of-the-Art)
├─ Add cutting-edge research in the field
└─ Ensure timeliness and relevance

Step 7: Build Citation Closure
└─ Establish a complete citation relationship network for all retrieved papers
```

### Technical Implementation
- **Seed Source**: arXiv API
- **Expansion Engine**: OpenAlex API
- **Citation Network**: Multi-level snowball sampling + Co-citation analysis

---

## 📄 Module 2: Paper Parsing and Deep Information Extraction

### Goal
- Parse PDFs to extract information section by section.
- Accurately extract deep semantic information (Problem, Contribution, Limitation, Future Work).

### Method: Multi-Agent Collaborative System

```text
┌─────────────────────────────────────────────┐
│           Multi-Agent Extraction            │
└─────────────────────────────────────────────┘
           ↓
    ┌──────────────┐
    │  Navigator   │  Locate Sections
    │    Agent     │  "Which section contains this info?"
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │  Extractor   │  Extract Sentences
    │    Agent     │  "Extract relevant content from sections"
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │   Critic     │  Quality Assessment
    │    Agent     │  "Is the result accurate? Re-extract?"
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Synthesizer  │  Summarize & Score
    │    Agent     │  "Generate final output and scoring"
    └──────────────┘
```

### Extraction Dimensions

| Dimension | Description | Target Sections |
| --- | --- | --- |
| **Problem-Contribution** | Problem-Method Pair | Abstract, Introduction |
| **Limitation** | Shortcomings/Defects | Discussion, Conclusion, Limitation |
| **Future Work** | Future research directions | Discussion, Conclusion, Future Work |

### Key Features
- **Section Awareness**: Automatically identifies paper structure.
- **Iterative Correction**: Critic-driven re-extraction mechanism.
- **Quality Assurance**: Every extracted result comes with a quality score.

---

## 🔗 Module 3: Citation Relationship Enrichment

### Goal
- Mine deep semantic relationships between papers.
- Prevent the confusion of different relationship types.

### Method: Socket Matching Mechanism
Treat the deep information of papers as **"Interfaces (Sockets)"**. The system uses LLMs and citation contexts to determine if these interfaces can be successfully connected.

### Logical Docking Matrix (4 Matches → 6 Relation Types)

```text
┌─────────────────────────────────────────────────────┐
│              Socket Matching Logic                  │
└─────────────────────────────────────────────────────┘

Match 1: Limitation → Problem
├─ Paper A's limitation → Paper B solved this problem
└─ Relation Type: Overcomes

Match 2: Future_Work → Problem
├─ Paper A's future work → Paper B realized this idea
└─ Relation Type: Realizes

Match 3: Problem → Problem (Under Same Method)
├─ Same method used to solve different problems
└─ Relation Type: Adapts_to (Transfer application)

Match 4: Method → Method (Under Same Problem)
├─ Same problem, different methods
├─ 4a. Extension in same direction → Extends
├─ 4b. Alternative in different direction → Alternative
└─ 4c. No clear explicit relation → Baselines

No Matches Found
└─ Relation Type: Baselines (Baseline comparison)
```

### Input Data
- Deep information of papers (Problem, Contribution, Limitation, Future_Work)
- Citation contexts
- LLM reasoning capabilities

### Output Results
- Each citation edge is annotated with one of the 6 relation types
- Relationship strength score
- Supporting evidence (citation context snippets)

---

## 🕸️ Module 4: Knowledge Graph Construction & Evolution Trajectory Mapping

### Goal
Fully utilize the deep information and citation relationships of papers to capture the deep logic of a literature review.
Papers serve as nodes and citation relationships as edges, integrating into a multi-dimensional knowledge graph. 
The goal is to find a small but highly refined set of key evolutionary trajectories from a massive pool of papers. The focus is on logical **"depth"** rather than superficial "breadth."

### Method
**Retain high-quality papers via strong-relation pruning, then identify three evolutionary patterns (chain, divergence, convergence) to generate a structured academic review report.**

**Step 1: Relation-Based Pruning**
- Traverse the entire graph using Bidirectional BFS to find strongly connected components (excluding `baselines` citation types).

**Step 2: Critical Evolutionary Paths Identification**
1. Select key nodes from the scoped connected components.
2. Identify three evolutionary patterns:
  - **Chain (A→B→C)**: Starting from a node within a specific scope, perform Depth-First Search (DFS) along strong relation edges to identify linear evolutionary chains.
  - **Divergence (Center ← Multiple branches)**: A single foundational paper sparks multiple branch development directions.
  - **Convergence (Center → Multiple baselines)**: Multiple independent prior works are integrated and built upon by a single subsequent paper.

---

## 💡 Module 5: Research Idea Generation

### Goal
- Generate novel and highly feasible research ideas.
- Automatically evaluate the quality of the generated ideas.

### Method: Defect Pool × Method Pool → Idea Combination

#### Step 1: Construct Fragment Pools (Based on Socket Matching)

```text
Pool A: Un-Overcome Limitations
├─ Filtered from Limitations of all papers
└─ Excludes those already solved via "Overcomes" relations

Pool B: Methods Extended ≥ 2 times
├─ Identifies successfully and repeatedly extended methods
└─ Indicates method versatility and transferability

Pool C: Methods from "Adapts_to"
├─ Methods successfully transferred to other domains
└─ Indicates cross-domain application potential

Pool D: Un-Realized Future Work
├─ Filtered from Future_Work of all papers
└─ Excludes those already achieved via "Realizes" relations
```

#### Step 2: Idea Generation (with Auto-Filtering)

```text
Cartesian Product Combination
    ↓
Limitation × Method → Candidate Ideas
    ↓
Chain of Thought (CoT) Reasoning
    ↓
┌────────────────────────────────────┐
│  1. Compatibility Analysis         │
│     Can the method solve the bug?  │
├────────────────────────────────────┤
│  2. Gap Identification             │
│     What improvements are needed?  │
├────────────────────────────────────┤
│  3. Idea Drafting                  │
│     Generate full research proposal│
└────────────────────────────────────┘
    ↓
Auto-Filtering: Keep only ideas with status="SUCCESS"
    ↓
Output High-Quality Idea List
```

---

## 🙏 Acknowledgments

This system is built upon the following open-source projects and data sources:
- [OpenAlex](https://openalex.org/) - Open catalog to the global research system
- [arXiv](https://arxiv.org/) - Preprint repository
- Excellent open-source libraries including NetworkX, Plotly, Sentence-Transformers, etc.

---

## 📧 Contact

For any questions or collaboration inquiries, please contact:
- **Email**: 1743623557@qq.com

---
**🎓 Let AI empower research innovation!**
