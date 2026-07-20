from __future__ import annotations


SAMPLE_DOCUMENTS = [
    "Motorcycle riders compare engine performance, helmets, tires, and road safety.",
    "Space researchers discuss NASA missions, orbital mechanics, telescopes, and astronomy.",
    "Hockey fans debate playoff scores, goalkeepers, teams, and the latest season.",
    "Computer graphics developers render images with GPUs, shaders, pixels, and 3D scenes.",
    "Medical professionals review symptoms, diagnosis, patient care, and treatment options.",
    "Car owners talk about engines, brakes, oil changes, and highway driving.",
]


def load_20_newsgroups(subset: str = "all", limit: int | None = None) -> tuple[list[str], list[str]]:
    """Load 20 Newsgroups when scikit-learn is installed, else sample docs."""
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError:
        documents = SAMPLE_DOCUMENTS[:limit]
        target_names = ["sample"]
        return documents, target_names

    dataset = fetch_20newsgroups(subset=subset)
    documents = dataset.data[:limit] if limit else dataset.data
    return list(documents), list(dataset.target_names)
