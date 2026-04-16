📚 Evonarrator: Evolutionary-Narrative–Driven Scientific Hypothesis Generation
Evonarrator is an end-to-end academic paper analysis system that closes the loop from literature retrieval to research idea generation by constructing citation knowledge graphs, performing deep information extraction, and mining semantic relationships across papers.

🌟 Core Features
The system supports the full research workflow through five tightly integrated modules:

Paper Retrieval and Citation Network Construction – a six-step snowballing strategy
Paper Downloading, Parsing, and Deep Information Extraction – a multi-agent extraction framework
Citation Relation Enrichment – socket-based semantic matching
Knowledge Graph Construction and Evolutionary Path Modeling – deep evolutionary narrative discovery
Research Idea Generation – creative synthesis based on limitation pools and method libraries
🔍 Module 1: Paper Retrieval and Citation Network Construction
Objective
Retrieve papers highly relevant to a given research topic
Construct a rich and well-connected citation network
Method: Six-Step Snowballing Retrieval Pipeline
<TEXT>
Step 1: Foundational Seeds
├─ Query arXiv to identify seminal and foundational papers as seed nodes
Step 2: Forward Snowballing
├─ Who cites the seed papers? → identify descendant nodes
└─ Retrieve citation links via the OpenAlex API
Step 3: Backward Snowballing
├─ Who is cited by the seed papers? → identify parent and ancestor nodes
└─ Trace the technical origins of the field
Step 4: Lateral Expansion / Co-citation Mining
├─ Which papers are repeatedly co-cited but missing from the corpus?
└─ Supplement overlooked but influential works
🔄 Step 5 (Optional): Second-Round Snowball Expansion
├─ Forward: expand from first-round papers
├─ Backward: trace their references
└─ Co-citation: analyze second-round co-citation patterns
Step 6: SOTA Completion
├─ Add the most recent state-of-the-art papers
└─ Ensure temporal coverage and freshness
Step 7: Citation Closure
└─ Build a complete citation graph across all collected papers
Technical Implementation
Seed Source: arXiv API
Expansion Engine: OpenAlex API
Citation Network: multi-layer snowballing + co-citation analysis
📄 Module 2: Paper Parsing and Deep Information Extraction
Objective
Parse PDF files into structured sections
Accurately extract deep semantic information (Problem, Contribution, Limitation, Future Work)
Method: Multi-Agent Collaborative Extraction
<TEXT>
┌─────────────────────────────────────────────┐
│            Multi-Agent Extraction            │
└─────────────────────────────────────────────┘
           ↓
    ┌──────────────┐
    │  Navigator   │  Locates sections
    │    Agent     │  "Which section contains this information?"
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │  Extractor   │  Extracts sentences
    │    Agent     │  "Extract relevant content from the section"
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │    Critic    │  Quality assessment
    │    Agent     │  "Is the extraction accurate? Re-extract if needed."
    └──────┬───────┘
           ↓
    ┌──────────────┐
    │ Synthesizer  │  Aggregation and scoring
    │    Agent     │  "Produce final output and assign confidence scores"
    └──────────────┘
Extracted Information Dimensions
Dimension	Description	Typical Sections
Problem–Contribution	Problem–method pairing	Abstract, Introduction
Limitation	Limitations and weaknesses	Discussion, Conclusion, Limitations
Future Work	Open directions and future research	Discussion, Conclusion, Future Work
Key Properties
Section-aware extraction: automatic recognition of paper structure
Iterative correction: critic-driven re-extraction loop
Quality control: each extracted item is associated with a confidence score
🔗 Module 3: Citation Relation Enrichment
Objective
Discover deep semantic relationships between papers
Avoid conflating different citation semantics
Method: Socket-Based Matching Mechanism
Each paper’s deep information is treated as a semantic “socket”, and citation relationships are inferred by determining whether sockets can be meaningfully connected using LLM reasoning and citation context.

Matching Logic Matrix (4 Matches → 6 Relation Types)
<TEXT>
┌─────────────────────────────────────────────────────┐
│               Socket Matching Logic                  │
└─────────────────────────────────────────────────────┘
Match 1: Limitation → Problem
├─ A limitation in Paper A is addressed by Paper B
└─ Relation: Overcomes
Match 2: Future Work → Problem
├─ A future direction proposed in Paper A is realized by Paper B
└─ Relation: Realizes
Match 3: Same Method, Different Problems
├─ A method applied to new problem settings
└─ Relation: Adapts_to
Match 4: Same Problem, Different Methods
├─ Alternative solutions to the same problem
├─ 4a. Same direction → Extends
├─ 4b. Different direction → Alternative
└─ 4c. No clear semantic relation → Baselines
No Match
└─ Relation: Baselines
Inputs
Deep paper information (Problem, Contribution, Limitation, Future Work)
Citation context
LLM-based reasoning
Outputs
One of six relation types per citation edge
Relation strength score
Supporting evidence (citation context snippets)
🕸️ Module 4: Knowledge Graph Construction and Evolutionary Path Modeling
Objective
Leverage deep semantic information and enriched citation relations to construct a multi-dimensional knowledge graph that captures the evolutionary logic of a research field.

Rather than maximizing breadth, the goal is to extract a small number of high-quality evolutionary narratives with strong logical coherence.

Method
High-quality paper selection via relation-based pruning, followed by identification of evolutionary patterns and structured survey generation.

Step 1: Relation-Based Graph Pruning
Traverse the graph and apply bidirectional BFS to identify strongly connected components
Retain only strong semantic relations (excluding baseline-only links)
Step 2: Critical Evolutionary Path Identification
Select key nodes within each connected component
Identify three evolutionary patterns:
Chains (A → B → C): linear evolutionary paths discovered via DFS over strong relations
Divergence: one paper giving rise to multiple development branches
Convergence: multiple independent prior works integrated by a single paper
💡 Module 5: Research Idea Generation
Objective
Generate novel and feasible research ideas
Automatically assess idea quality
Method: Limitation Pool × Method Library → Idea Synthesis
Step 1: Fragment Pool Construction (Based on Socket Matching)
<TEXT>
Pool A: Unresolved Limitations
├─ Limitations not addressed by any Overcomes relation
Pool B: Frequently Extended Methods
├─ Methods extended by two or more papers
└─ Indicative of robustness and generality
Pool C: Adapted Methods
├─ Methods successfully transferred across domains
└─ High cross-domain potential
Pool D: Unrealized Future Work
├─ Future directions not realized by any Realizes relation
Step 2: Idea Generation and Filtering
<TEXT>
Cartesian product
    ↓
Limitation × Method → Candidate ideas
    ↓
Chain-of-Thought reasoning
    ↓
┌────────────────────────────────────┐
│  1. Compatibility Analysis          │
│     Can the method address the gap? │
├────────────────────────────────────┤
│  2. Gap Identification              │
│     What extensions or changes are  │
│     required?                       │
├────────────────────────────────────┤
│  3. Idea Drafting                   │
│     Generate a complete proposal    │
└────────────────────────────────────┘
    ↓
Automatic filtering (status = "SUCCESS")
    ↓
Final high-quality idea list
🙏 Acknowledgements
This system is built upon the following open-source projects and data sources:

OpenAlex – Open scholarly metadata API
arXiv – Open-access preprint repository
NetworkX, Plotly, Sentence-Transformers, and other open-source libraries
📧 Contact
For questions or collaboration, please contact:

Email: 1743623557@qq.com
🎓 Empowering scientific discovery with AI.
