def super_fancy_loading_animation():
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    phrases = [
        "Analyzing market insights",
        "Optimizing solutions",
        "Processing business intelligence",
        "Generating recommendations",
        "Evaluating opportunities",
        "Synthesizing data points",
        "Calculating outcomes",
        "Enhancing business value",
        "Streamlining workflows",
        "Maximizing efficiency"
    ]
    
    spinner = cycle(frames)
    phrase_index = 0
    fade_counter = 0
    typing_index = 0
    i = 0
    
    while not future.done():
        frame = next(spinner)
        current_phrase = phrases[phrase_index]
        
        # Get current display text
        if typing_index <= len(current_phrase):
            displayed_text = current_phrase[:typing_index]
            dots = ""
        else:
            displayed_text = current_phrase
            dots = "." * min(typing_index - len(current_phrase), max_dots)
        
        # Increment typing index
        if typing_index < len(current_phrase) + max_dots:
            typing_index += 1
        
        # Handle phrase transition
        if fade_counter >= 15:  # 1.5 seconds display time
            fade_counter = 0
            phrase_index = (phrase_index + 1) % len(phrases)
            typing_index = 0  # Reset typing index for new phrase
        
        fade_counter += 1
        
        gradient_text = f"""
        <div style="
            font-size: 24px;
            font-family: monospace;
            padding: 3px 0;
            text-align: left;
            display: flex;
            align-items: center;
            gap: 8px;
            ">
            <span style="color: rgb(55, 65, 81);">{displayed_text}{dots}</span>
            <span style="color: rgb(255, 75, 75);">{frame}</span>
        </div>
        """
        
        dots_area.markdown(gradient_text, unsafe_allow_html=True)
        time.sleep(0.05)



"""
/* User message specific styling */
        .user-message {
            background-color: #2b313e;
            border-radius: 15px;
            padding: 10px 15px;
            margin-left: auto;  /* Push message to the right */
            margin-right: 0;
            max-width: fit-content;  /* Adjust width to content */
            text-align: right;
        }
"""
