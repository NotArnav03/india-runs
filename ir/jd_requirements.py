"""Curated requirement model for the single released JD.

Gap analysis item C, reshaped: with exactly one (deliberately adversarial)
job description, the high-value work is not a generic NLP extractor but an
*auditable* encoding of what this JD says and means — the kind of artifact a
Stage-5 interviewer can be walked through line by line.

Every field below is traceable to ``job_description.docx``.  Quotes from the
JD are kept in the comments so the provenance is explicit.  Nothing here is
guessed: the JD is unusually direct about its must-haves, nice-to-haves, and
— rare for a JD — its hard *disqualifiers* and its "read between the lines"
ideal profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ir.features import Job


@dataclass(frozen=True)
class JDRequirements:
    """Structured, auditable model of one job description."""

    title: str
    # The text the candidate profiles are semantically matched against.  We
    # use a distilled "what we actually need" paragraph rather than the full
    # rambling JD, so the embedding centers on the real signal.
    semantic_target: str

    # Skills the JD says you "absolutely need" vs. "would like".
    must_have_skills: frozenset[str]
    nice_to_have_skills: frozenset[str]

    # Experience band — explicitly "a range, not a requirement".
    experience_band: tuple[float, float]
    experience_hard_floor: float  # below this, even strong signals can't fully save it

    # Career-evidence phrases that indicate the JD's true target: someone who
    # *shipped* retrieval / ranking / search / recsys at a product company.
    shipped_systems_phrases: frozenset[str]

    # Hard disqualifiers (JD: "the disqualifiers we actually apply" + "things
    # we explicitly do NOT want").
    consulting_only_firms: frozenset[str]
    research_only_terms: frozenset[str]
    out_of_domain_terms: frozenset[str]  # CV / speech / robotics w/o NLP/IR
    framework_only_terms: frozenset[str]  # LangChain-wrapper-only "AI experience"

    # Logistics.
    target_locations: frozenset[str]
    preferred_notice_max_days: int

    # Roles that signal a wrong-fit profile even when the skill list looks AI-heavy
    # (JD's "Marketing Manager with all the keywords is not a fit" trap).
    off_target_title_terms: frozenset[str]

    def build_job(self) -> Job:
        """Construct the schema-independent :class:`Job` for the ranker."""
        return Job(
            id="redrob_senior_ai_engineer",
            text=self.semantic_target,
            required_skills=set(self.must_have_skills),
            metadata={"requirements": self},
        )


# ─── The released JD, encoded ─────────────────────────────────────────────

REDROB_SENIOR_AI_ENGINEER = JDRequirements(
    title="Senior AI Engineer — Founding Team (Redrob AI)",
    # Distilled from the JD's "What we'd actually be doing" + "Things you
    # absolutely need" + "How to read between the lines".  Phrased as the
    # ideal candidate's own profile would read, to maximize semantic overlap
    # with genuine fits.
    semantic_target=(
        "Senior AI / ML engineer who owns the intelligence layer of a product: "
        "ranking, retrieval, and matching systems in production. Hands-on with "
        "embeddings-based retrieval (sentence-transformers, BGE, E5, OpenAI "
        "embeddings) deployed to real users, handling embedding drift, index "
        "refresh, and retrieval-quality regression. Production experience with "
        "vector databases and hybrid search (Pinecone, Weaviate, Qdrant, Milvus, "
        "FAISS, Elasticsearch, OpenSearch). Strong Python and code quality. "
        "Designs rigorous evaluation frameworks for ranking systems: NDCG, MRR, "
        "MAP, offline-to-online correlation, A/B testing. Has shipped at least "
        "one end-to-end ranking, search, or recommendation system to real users "
        "at meaningful scale at a product company. Scrappy product-engineering "
        "mindset, ships fast, writes well, works async. 6 to 8 years of "
        "experience with 4 to 5 in applied ML at product companies."
    ),
    must_have_skills=frozenset({
        # JD "Things you absolutely need"
        "embeddings", "sentence-transformers", "retrieval", "information retrieval",
        "vector database", "vector search", "hybrid search", "semantic search",
        "faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
        "opensearch", "python", "ranking", "learning to rank", "ndcg", "mrr",
        "map", "ranking evaluation", "recommendation", "recommender systems",
        "nlp", "rag",
    }),
    nice_to_have_skills=frozenset({
        # JD "Things we'd like you to have but won't reject you for"
        "lora", "qlora", "peft", "fine-tuning", "xgboost", "lightgbm",
        "learning-to-rank", "hr-tech", "recruiting", "marketplace",
        "distributed systems", "inference optimization", "open source",
        "pytorch", "transformers",
    }),
    experience_band=(5.0, 9.0),
    experience_hard_floor=3.0,
    shipped_systems_phrases=frozenset({
        "ranking", "retrieval", "search", "recommendation", "recommender",
        "recsys", "embedding", "vector", "relevance", "matching", "personalization",
        "information retrieval", "semantic", "hybrid search", "learning to rank",
        "nearest neighbor", "candidate generation",
    }),
    # JD: "People who have only worked at consulting firms (TCS, Infosys,
    # Wipro, Accenture, Cognizant, Capgemini, etc.)".
    consulting_only_firms=frozenset({
        "tcs", "tata consultancy", "infosys", "wipro", "accenture",
        "cognizant", "capgemini", "hcl", "tech mahindra", "mindtree",
        "ltimindtree", "deloitte", "ibm global services", "dxc",
    }),
    # JD: "pure research environments (academic labs, research-only roles)".
    research_only_terms=frozenset({
        "research scientist", "research fellow", "phd researcher", "postdoc",
        "research assistant", "academic", "research intern", "research lab",
    }),
    # JD: "primary expertise is computer vision, speech, or robotics without
    # significant NLP/IR exposure".
    out_of_domain_terms=frozenset({
        "computer vision", "image classification", "object detection",
        "speech recognition", "robotics", "autonomous", "lidar", "slam",
    }),
    # JD: "'AI experience' consists primarily of recent (under 12 months)
    # projects using LangChain to call OpenAI".
    framework_only_terms=frozenset({
        "langchain", "llamaindex", "autogpt", "prompt engineering",
    }),
    # JD: "Located in or willing to relocate to Noida or Pune"; also
    # "Hyderabad, Pune, Mumbai, Delhi NCR welcome".
    target_locations=frozenset({
        "noida", "pune", "delhi", "new delhi", "gurgaon", "gurugram",
        "ghaziabad", "faridabad", "delhi ncr", "ncr", "hyderabad", "mumbai",
    }),
    preferred_notice_max_days=30,  # JD: "We'd love sub-30-day notice."
    off_target_title_terms=frozenset({
        "marketing", "sales", "recruiter", "hr manager", "human resources",
        "content writer", "accountant", "designer", "customer success",
        "business development", "operations manager", "project manager",
    }),
)
