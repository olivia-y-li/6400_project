
#load hotpotqa distractor validation split
#randomly select 500 eval questions and 50 seperate development questions
#use fixed random seed for reproducibility

from datasets import load_dataset

def load_hotpotqa(num_eval=500, num_dev=50, seed=42):

    # Load HotpotQA distractor validation split
    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="validation"
    )

    # Keep only hard questions
    dataset = dataset.filter(lambda x: x["level"] == "hard")

    # Randomly shuffle with a fixed seed
    dataset = dataset.shuffle(seed=seed)

    # Select 550 questions
    dataset = dataset.select(range(num_eval + num_dev))

    # Split into development and evaluation sets
    dev_data = dataset.select(range(num_dev))
    eval_data = dataset.select(range(num_dev, num_dev + num_eval))

    return dev_data, eval_data


if __name__ == "__main__":
    dev_data, eval_data = load_hotpotqa()

    print("Development questions:", len(dev_data))
    print("Evaluation questions:", len(eval_data))

    # Inspect one example
    example = dev_data[0]

    print("Question:", example["question"])
    print("Answer:", example["answer"])
    print("Supporting facts:", example["supporting_facts"])
    print("Context:", example["context"])