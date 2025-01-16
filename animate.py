import time
from itertools import cycle

def fancy_loading_animation():
    # Cool spinner frames using block elements
    frames = [
        "▰▱▱▱▱▱▱▱",
        "▰▰▱▱▱▱▱▱",
        "▰▰▰▱▱▱▱▱",
        "▰▰▰▰▱▱▱▱",
        "▰▰▰▰▰▱▱▱",
        "▰▰▰▰▰▰▱▱",
        "▰▰▰▰▰▰▰▱",
        "▰▰▰▰▰▰▰▰",
        "▰▰▰▰▰▰▰▱",
        "▰▰▰▰▰▰▱▱",
        "▰▰▰▰▰▱▱▱",
        "▰▰▰▰▱▱▱▱",
        "▰▰▰▱▱▱▱▱",
        "▰▰▱▱▱▱▱▱",
        "▰▱▱▱▱▱▱▱",
        "▱▱▱▱▱▱▱▱"
    ]
    
    # Words that will appear with a typing effect
    words = [
        "INITIALIZING SYSTEM",
        "PROCESSING REQUEST",
        "ANALYZING DATA",
        "PLEASE STAND BY"
    ]
    
    spinner = cycle(frames)
    i = 0
    word_idx = 0
    
    while not future.done():
        current_word = words[word_idx]
        displayed_word = current_word[:((i//2) % (len(current_word) + 1))]
        frame = next(spinner)
        
        # Create a dynamic multi-line display
        display = f"""
