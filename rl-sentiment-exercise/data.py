"""
Dataset module for RL sentiment fine-tuning exercise.

Provides prompts for training a language model to generate positive sentiment text.
"""

from datasets import Dataset


# QUESTION Q1: In standard RL, the "environment" provides states and rewards.
# In our LLM fine-tuning setup, what is the environment - i.e. what are the "states" and "actions"?

# Movie review starter prompts for sentiment generation
TRAIN_PROMPTS = [
    # Neutral starters
    "This movie was",
    "The film was",
    "I watched this and",
    "The acting was",
    "The story was",
    "The plot was",
    "Overall, I think",
    "In my opinion,",
    "The director did",
    "This film made me",
    "The cinematography was",
    "The soundtrack was",
    "The characters were",
    "The ending was",
    "The beginning of the movie",
    "The middle part was",
    "I found the movie",
    "The screenplay was",
    "The dialogue was",
    "The pacing was",
    # Slightly more context
    "After watching this movie, I",
    "When the credits rolled, I",
    "The lead actor's performance was",
    "The supporting cast was",
    "The special effects were",
    "The emotional depth was",
    "The humor in this film",
    "The action sequences were",
    "The romance subplot was",
    "The villain was",
    "The hero's journey was",
    "The twist at the end",
    "The opening scene was",
    "The final act was",
    "The chemistry between leads",
    "This adaptation was",
    "Compared to the book,",
    "The sequel was",
    "The prequel was",
    "This remake was",
    # Genre-specific
    "For a comedy, this",
    "As a drama, this",
    "This thriller was",
    "The horror elements were",
    "The sci-fi aspects were",
    "The fantasy world was",
    "This documentary was",
    "The animation was",
    "This musical was",
    "The romantic comedy was",
]

# Separate validation prompts (64 prompts, out-of-sample from training)
VALIDATION_PROMPTS = [
    # General reactions
    "My experience watching this was",
    "The movie surprised me by",
    "I would describe this film as",
    "The best part was",
    "What stood out was",
    "The film's message was",
    "This movie is worth",
    "I recommend this because",
    "The production quality was",
    "The performances were",
    # Emotional responses
    "This movie left me feeling",
    "I was moved by",
    "The emotional impact was",
    "I laughed when",
    "I cried during",
    "The tension built up",
    "I felt connected to",
    "The atmosphere was",
    # Technical aspects
    "The camera work was",
    "The editing was",
    "The lighting created",
    "The sound design was",
    "The visual effects added",
    "The costume design was",
    "The set design was",
    "The makeup and styling",
    # Story elements
    "The narrative structure was",
    "The character development was",
    "The plot progression was",
    "The backstory revealed",
    "The subplots were",
    "The themes explored",
    "The symbolism in this film",
    "The moral of the story",
    # Comparisons
    "Compared to similar films,",
    "This stands out because",
    "Unlike other movies,",
    "This reminded me of",
    "In the genre, this",
    "Among recent releases,",
    "For its budget,",
    "Given the hype,",
    # Specific scenes
    "The climax was",
    "The resolution felt",
    "The introduction set up",
    "The turning point was",
    "The reveal was",
    "The confrontation scene",
    "The quiet moments were",
    "The action scenes delivered",
    # Audience perspective
    "I would tell others that",
    "Viewers should expect",
    "This appeals to",
    "Fans of the genre will",
    "Casual viewers might",
    "Critics would say",
    "The target audience for",
    "Rewatching this would",
    # Overall assessment
    "On reflection, I think",
    "The lasting impression is",
    "In summary, this film",
    "My final verdict is",
    "All things considered,",
    "Looking back, I feel",
]

# QUESTION Q2: How do the prompts differ from typical modern LLM prompts?
# Hint: notice that the exercise relies on GPT-2, which was not tuned specifically for questions or instructions.

def get_train_dataset() -> Dataset:
    """
    Create the training dataset with prompts.
    
    Returns:
        Dataset with 'prompt' column for TRL GRPOTrainer.
    """
    return Dataset.from_dict({"prompt": TRAIN_PROMPTS})


def get_validation_dataset() -> Dataset:
    """
    Create the validation dataset with prompts for evaluation.
    
    These 64 prompts are out-of-sample from training prompts.
    
    Returns:
        Dataset with 'prompt' column.
    """
    return Dataset.from_dict({"prompt": VALIDATION_PROMPTS})


def get_combined_dataset() -> Dataset:
    """
    Create a combined dataset with all prompts.
    
    Returns:
        Dataset with 'prompt' column containing both train and validation prompts.
    """
    all_prompts = TRAIN_PROMPTS + VALIDATION_PROMPTS
    return Dataset.from_dict({"prompt": all_prompts})


if __name__ == "__main__":
    # Quick test
    train_ds = get_train_dataset()
    val_ds = get_validation_dataset()
    
    print(f"Training prompts: {len(train_ds)}")
    print(f"Validation prompts: {len(val_ds)}")
    print("\nSample training prompts:")
    for i in range(min(5, len(train_ds))):
        print(f"  - {train_ds[i]['prompt']}")
    print("\nSample validation prompts:")
    for i in range(min(5, len(val_ds))):
        print(f"  - {val_ds[i]['prompt']}")
