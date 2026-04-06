import os

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

OUT_PATH = "data/paloma/tokenized_paloma_c4_100_domains_validation.bin"

SYNTHETIC_TEXTS = [
    "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.",
    "Computer science is the study of computation, algorithms, data structures, programming languages, and computer hardware. The field was founded in the 1950s and draws heavily from mathematics, electrical engineering, and linguistics.",
    "Mathematics is the study of numbers, shapes, and patterns. It includes the study of pure mathematics such as algebra, geometry, analysis, and number theory, as well as applied mathematics in physics, engineering, and finance.",
    "Physics is the natural science that studies matter, its motion and behavior through space and time, and the related entities of energy and force. Physics is one of the most fundamental scientific disciplines.",
    "Chemistry is the scientific discipline involved with elements and compounds composed of atoms, molecules and ions: their composition, structure, properties, behavior and the changes they undergo during a reaction with other substances.",
    "Biology is the scientific study of life. It is a natural science with a broad scope but has several unifying themes that tie it together as a single, coherent field. For instance, all organisms are made up of cells that process hereditary information encoded in genes.",
    "History is the study and the documentation of the past. Events before the invention of writing systems are considered prehistory. History is an umbrella term comprising past events as well as the memory, discovery, collection, organization, presentation, and interpretation of information about these events.",
    "Philosophy is the study of general and fundamental questions about existence, knowledge, values, reason, mind, and language. Such questions are often posed as problems to be studied or resolved.",
    "Economics is the social science that studies the production, distribution, and consumption of goods and services. Economics focuses on the behavior and interactions of economic agents and how economies work.",
    "Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. Statistical methods are used in all kinds of applications, both in science and industry.",
    "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
    "An algorithm is a finite sequence of well-defined, computer-implementable instructions, typically to solve a class of problems or to perform a computation.",
    "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from noisy, structured and unstructured data.",
    "Probability theory is the branch of mathematics concerned with probability. Although there are several different probability interpretations, probability theory treats the concept in a rigorous mathematical manner.",
    "Calculus is the mathematical study of continuous change. It has two major branches, differential calculus and integral calculus.",
    "Linear algebra is the branch of mathematics concerning linear equations and linear maps and their representations in vector spaces and through matrices.",
    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
    "Democracy is a form of government in which the people have the authority to deliberate and decide legislation, or to choose governing officials to do so.",
    "Evolution is change in the heritable characteristics of biological populations over successive generations. These characteristics are the expressions of genes that are passed on from parent to offspring during reproduction.",
    "The solar system consists of the Sun and the objects that orbit it, either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight planets.",
    "Climate change includes both global warming driven by human-induced emissions of greenhouse gases and the resulting large-scale shifts in weather patterns.",
    "The human brain is the command center for the human nervous system. It receives signals from the body's sensory organs and outputs information to the muscles.",
    "Language is a structured system of communication used by humans. The scientific study of language is called linguistics.",
    "The history of the Internet began with the development of electronic computers in the 1950s.",
    "Astronomy is a natural science that studies celestial objects and phenomena using mathematics, physics, and chemistry.",
    "Medicine is the science and practice of caring for a patient, managing the diagnosis, prognosis, prevention, treatment, and palliation of their injury or disease.",
    "Psychology is the scientific study of mind and behavior, including feelings and thoughts.",
    "Sociology is a social science that focuses on society, human social behavior, and patterns of social relationships.",
]


def build_synthetic(tokenizer: "AutoTokenizer", n_docs: int = 2000) -> list[int]:
    """Generate synthetic token stream when real Paloma is unavailable."""
    eos = tokenizer.eos_token_id
    all_ids: list[int] = []
    for i in range(n_docs):
        text = SYNTHETIC_TEXTS[i % len(SYNTHETIC_TEXTS)]
        text = f"{text} (Document {i + 1} of synthetic Paloma fallback.)"
        all_ids.extend(tokenizer.encode(text) + [eos])
    return all_ids


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    all_ids: list[int] = []

    try:
        from datasets import load_dataset

        print("Attempting to download allenai/paloma from HuggingFace...")
        try:
            ds = load_dataset("allenai/paloma", "c4_100_domains", split="validation")
        except ValueError:
            ds = load_dataset("allenai/paloma", "c4_100_domains", split="val")

        for item in tqdm(ds, desc="Tokenizing Paloma"):
            all_ids.extend(tokenizer.encode(item["text"]) + [tokenizer.eos_token_id])
        print(f"Downloaded and tokenized {len(ds)} real Paloma documents.")

    except Exception as e:
        print(f"Could not load real Paloma ({e}).")
        print("Falling back to synthetic Paloma data (2000 documents).")
        all_ids = build_synthetic(tokenizer)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.array(all_ids, dtype=np.uint16).tofile(OUT_PATH)
    print(f"Saved {len(all_ids)} tokens to {OUT_PATH}")


if __name__ == "__main__":
    main()
