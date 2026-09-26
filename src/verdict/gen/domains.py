"""What to generate for a domain: the label space, and the axes that keep a small local
model from writing the same message a thousand times."""
from __future__ import annotations

from dataclasses import dataclass


LEDGER_INTRO = "You are writing one message that {author} posts in a shared engineering room about: {topic}."


@dataclass(frozen=True)
class Domain:
    name: str
    types: dict[str, str]  # intended type -> what a message of that type does
    confusable: dict[str, str]  # type -> the type it is most often mistaken for
    room_topics: tuple[str, ...]  # topics / situations
    authors: tuple[str, ...]  # authors / personas
    lengths: tuple[str, ...]
    intro: str = LEDGER_INTRO
    lead: str = "The message"  # "<lead> <type meaning>."
    style: str = (
        "Use concrete, specific details (names, numbers, file or service names). "
        "Do not name the kind of message it is, and do not use headings."
    )
    ambiguity_lead: str = "a message that"
    state_style: str = "ledger"  # "ledger" -> {room_topic, author, message}; "text" -> {"text": message}


TYPES = {
    "action": "reports work that was done or shipped, with concrete results such as commits, files, tests run or numbers measured",
    "synthesis": "compiles the conclusions of a discussion into a reference: what is settled, what is open, current state of the room",
    "decision": "records a choice that was made between options, with the rationale for it",
    "thought": "exploratory reasoning: a hypothesis, a worry or an observation that is not yet settled",
    "draft": "a proposal or piece of writing put forward for feedback, not yet final",
    "note": "a journal-style observation worth keeping that is not part of an ongoing deliberation",
    "plan": "a specified sequence of work that is waiting to be executed, with steps and owners or order",
    "critique": "pushback on a proposal or approach: concerns, risks and what is wrong with it",
    "review": "feedback on someone else's finished work, with what is good and what to change",
    "code": "a message whose substance is a code or config snippet, with only a line or two of framing",
}

HUB_TYPES = tuple(TYPES)

MESSAGE_TYPING = Domain(
    name="hub_message_typing",
    types=TYPES,
    confusable={
        "thought": "note",
        "note": "thought",
        "draft": "plan",
        "plan": "draft",
        "critique": "review",
        "review": "critique",
        "action": "synthesis",
        "synthesis": "decision",
        "decision": "synthesis",
        "code": "action",
    },
    room_topics=(
        "Migrate the sync worker from Redis to Postgres",
        "Flaky integration tests on the payment service",
        "Add rate limiting to the public API",
        "Fine-tune a small classifier on support tickets",
        "Next.js app router migration",
        "Go module layout for the CLI",
        "Elixir supervision tree redesign",
        "Docker image size and build cache",
        "OAuth token refresh bug in the mobile app",
        "Benchmarking two embedding models for search",
        "Cloudflare Workers cold start latency",
        "Terraform state drift on staging",
        "Vector store choice for the RAG pipeline",
        "Accessibility audit of the dashboard",
        "Rename the public package before first release",
        "Postgres index bloat on the events table",
        "Design tokens and dark mode",
        "Grant application draft for the open-source fund",
        "Scraper rate limits and proxy rotation",
        "Release checklist for v1.0",
        "Reproducing a paper's ablation results",
        "Log redaction so secrets never reach disk",
        "Webhook signature verification",
        "Python 3.11 pin because of missing wheels",
        "Onboarding flow drop-off analysis",
        "Cost of the nightly batch job on Modal",
        "Choosing a license for the dataset",
        "Deprecating the v1 endpoints",
        "Kubernetes pod evictions on the batch cluster",
        "Feature flag cleanup before the release",
        "Slow CI: caching node_modules and Go build artifacts",
        "Search relevance regression after the tokenizer change",
        "GDPR data export request handling",
        "Migrating from REST to gRPC between two services",
        "Mobile app crash on Android 15 startup",
        "Prompt injection guard for the support agent",
        "Nightly backup verification and restore drill",
        "Choosing between SQLite and Postgres for the local cache",
        "Structured logging fields and correlation ids",
        "Dependency update that breaks the type checker",
        "Rate-limit headers on the public API",
        "LLM cost dashboard and per-team budgets",
        "Websocket reconnect storms after a deploy",
        "Monorepo build graph and affected-only tests",
        "Redesigning the onboarding checklist",
        "Embedding model swap and reindexing the vector store",
        "On-call runbook for the payments queue",
        "Timezone bugs in the scheduler",
        "Secrets rotation without downtime",
        "Image resize service memory leak",
        "Accessibility of the date picker",
        "Deprecating a CLI flag with a migration path",
        "Load test results for the checkout endpoint",
        "Fine-tuning dataset licence and provenance",
        "Flaky end-to-end test on the login flow",
        "Migrating the docs site to a new static generator",
        "Cold-start latency of the serverless functions",
        "Retention policy for audit logs",
        "Schema migration ordering across environments",
        "Evaluating a smaller model for classification",
    ),
    authors=("claude-code", "claude", "codex", "gemini-cli", "amp", "human"),
    lengths=(
        "one or two sentences",
        "a short paragraph of three to five sentences",
        "a paragraph followed by a short bullet list",
        "a single terse line",
    ),
)


EMOTION_TYPES = {
    "sadness": "feels sad, down, hurt or grieving",
    "joy": "feels joyful, happy, content or excited",
    "love": "feels love, affection, tenderness or caring for someone",
    "anger": "feels angry, annoyed, irritated or resentful",
    "fear": "feels afraid, anxious, nervous or threatened",
    "surprise": "feels surprised, amazed or shocked by something unexpected",
}

EMOTION = Domain(
    name="emotion",
    types=EMOTION_TYPES,
    confusable={"joy": "love", "love": "joy", "sadness": "fear", "fear": "sadness", "anger": "fear", "surprise": "joy"},
    room_topics=(
        "a job interview", "a family dinner", "the weather this weekend", "a long commute", "exam results",
        "a friend's wedding", "moving to a new city", "a pet", "a video game", "a concert", "a breakup",
        "a new baby", "money worries", "a doctor's appointment", "a football match", "a birthday party",
        "a noisy neighbour", "a delayed flight", "cooking dinner", "a call from an old friend", "a promotion",
        "being stuck in traffic", "a school play", "a hospital visit", "an argument with a sibling",
        "a surprise visit", "a lost wallet", "finishing a marathon", "a new apartment", "a group project",
        "a late night alone", "a family reunion", "a first date", "an unexpected gift", "a long shift",
        "a storm at night", "a graduation", "a rude customer", "an old photograph", "a job rejection",
        "a road trip", "a quiet morning", "a diagnosis", "a bad review", "coming home after a long trip",
    ),
    authors=("a teenager", "a college student", "a new parent", "a nurse on night shift", "a retired teacher",
             "a software developer", "a gamer", "a barista", "a long-distance runner", "someone living alone",
             "a football fan", "a shop owner"),
    lengths=("a fragment of a sentence", "one short sentence", "two short sentences", "one sentence of about twenty words"),
    intro="You are writing one short, casual social media post by {author}, about: {topic}.",
    lead="The writer",
    style=(
        "Write it the way people really post: first person, casual, often lowercase, sometimes a typo or missing "
        "punctuation, no hashtags, no emojis, no quotation marks. About half the time start with 'i feel', "
        "'i am feeling' or 'i felt'. Do not name the emotion in a way that sounds like a label."
    ),
    ambiguity_lead="a post where the writer",
    state_style="text",
)


#: GitHub issue type, the three labels of the NLBSE issue-report benchmark. The descriptions push
#: the generator toward the hard case FINDINGS §41 found: a question that reads like a bug report.
ISSUE_TYPES = {
    "bug": "reports that something is broken, crashes, errors or behaves wrongly, usually with steps "
           "to reproduce, an error message, or the versions involved",
    "feature": "asks for a new feature, an option, or a change to how something works",
    "question": "asks how to do something, why something behaves as it does, or for help "
                "understanding it; it may quote an error or code the writer tried, but it does not "
                "claim the project is broken",
}

ISSUES = Domain(
    name="github_issue_typing",
    types=ISSUE_TYPES,
    confusable={"question": "bug", "bug": "question", "feature": "question"},
    # Projects outside the NLBSE test set (react, tensorflow, vscode, bitcoin, opencv), so a gain
    # cannot come from learning those repositories' vocabulary.
    room_topics=(
        "Django, a Python web framework", "pandas, a dataframe library", "Kubernetes",
        "the Rust compiler", "Next.js", "PostgreSQL", "Home Assistant", "Flutter", "Terraform",
        "FastAPI", "Electron", "the Go standard library", "scikit-learn", "Prometheus", "Grafana",
        "Neovim", "Rails", "Spring Boot", "Svelte", "Tauri", "Deno", "Bun", "Pydantic", "Celery",
        "Redis", "Elasticsearch", "Ansible", "Jupyter", "Hugging Face transformers", "PyTorch Lightning",
        "Vite", "Tailwind CSS", "Supabase", "Prisma", "NumPy", "Matplotlib", "Kafka", "Airflow",
        "Docker Compose", "Helm", "Godot", "Blender", "Obsidian plugins", "Home-brew formulae",
        "the Kotlin compiler", "SwiftUI tooling", "gRPC", "SQLAlchemy", "Poetry", "pytest",
    ),
    authors=(
        "a first-time contributor", "a maintainer of a downstream library", "a data scientist",
        "a student learning the project", "a DevOps engineer", "a mobile developer",
        "a backend engineer in a hurry", "a hobbyist", "an enterprise user on an old version",
        "a non-native English speaker", "a security researcher", "a documentation writer",
    ),
    lengths=(
        "a title and a two-sentence body",
        "a title and a short paragraph",
        "a title and a body with a short code or log snippet",
        "a title and a body that follows an issue template with a few headings",
    ),
    intro="You are writing one GitHub issue opened by {author} on the repository of {topic}.",
    lead="The issue",
    style=(
        "Put the title on the first line and the body after it. Use concrete details: versions, "
        "file names, commands, error messages. Do not put a label or prefix such as 'Bug:', "
        "'[Feature]' or 'Question:' in the title, and do not say what kind of issue it is."
    ),
    ambiguity_lead="an issue that",
    state_style="issue",
)
