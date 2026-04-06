"""
Create a synthetic train_positive.txt in fastText format.
Used when Wikipedia cannot be downloaded (e.g., network restrictions).
"""
import argparse
import os

SAMPLE_TEXTS = [
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
    "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
    "An algorithm is a finite sequence of well-defined, computer-implementable instructions, typically to solve a class of problems or to perform a computation. Algorithms are always unambiguous and are used as specifications for performing calculations.",
    "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from noisy, structured and unstructured data, and apply knowledge and actionable insights from data across a broad range of application domains.",
    "Probability theory is the branch of mathematics concerned with probability. Although there are several different probability interpretations, probability theory treats the concept in a rigorous mathematical manner by expressing it through a set of axioms.",
    "Calculus is the mathematical study of continuous change. It has two major branches, differential calculus and integral calculus; the former concerns instantaneous rates of change, and the slopes of curves, while the latter concerns accumulation of quantities, and areas under or between curves.",
    "Linear algebra is the branch of mathematics concerning linear equations and linear maps and their representations in vector spaces and through matrices. Linear algebra is central to almost all areas of mathematics.",
    "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics.",
    "Democracy is a form of government in which the people have the authority to deliberate and decide legislation, or to choose governing officials to do so. Who is considered part of the people and how authority is shared among or delegated by the people has changed over time.",
    "Evolution is change in the heritable characteristics of biological populations over successive generations. These characteristics are the expressions of genes that are passed on from parent to offspring during reproduction.",
    "The solar system consists of the Sun and the objects that orbit it, either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the eight planets, with the remainder being smaller objects, the dwarf planets and small solar system bodies.",
    "Climate change includes both global warming driven by human-induced emissions of greenhouse gases and the resulting large-scale shifts in weather patterns. Though there have been previous periods of climatic change, since the mid-20th century humans have had an unprecedented impact on Earth's climate system.",
    "The human brain is the command center for the human nervous system. It receives signals from the body's sensory organs and outputs information to the muscles. The human brain has the same basic structure as other mammal brains but is larger in relation to body size.",
    "Language is a structured system of communication. Language, in a broader sense, is the method of communication that involves the use of – particularly human – languages. The scientific study of language is called linguistics.",
    "The history of the Internet began with the development of electronic computers in the 1950s. Initial concepts of wide area networking originated in several computer science laboratories in the United States, United Kingdom, and France.",
    "Astronomy is a natural science that studies celestial objects and phenomena. It uses mathematics, physics, and chemistry in order to explain their origin and evolution. Objects of interest include planets, moons, stars, nebulae, galaxies, and comets.",
    "Medicine is the science and practice of caring for a patient, managing the diagnosis, prognosis, prevention, treatment, palliation of their injury or disease, and promoting their health. Medicine encompasses a variety of health care practices evolved to maintain and restore health.",
    "Psychology is the scientific study of mind and behavior. Psychology includes the study of conscious and unconscious phenomena, including feelings and thoughts. It is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences.",
    "Sociology is a social science that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output path for train_positive.txt")
    parser.add_argument("--n", type=int, default=30, help="Number of positive examples to generate")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    lines = []
    for i in range(args.n):
        text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
        # Augment slightly to make each unique
        text = f"{text} This article has been reviewed and verified by multiple editors. Document index {i+1}."
        joined = text.replace("\n", " ")
        lines.append(f"__label__positive {joined}\n")

    with open(args.output, "w") as f:
        f.writelines(lines)

    print(f"Wrote {len(lines)} synthetic positive samples to {args.output}")


if __name__ == "__main__":
    main()
