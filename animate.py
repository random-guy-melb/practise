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
        max_dots = 3
        
        # Handle typing effect including dots
        if typing_index < len(current_phrase) + max_dots:
            typing_index += 1
        
        # Handle phrase transition
        if fade_counter >= 15:  # 1.5 seconds display time
            fade_counter = 0
            phrase_index = (phrase_index + 1) % len(phrases)
            typing_index = 0  # Reset typing index for new phrase
            
        fade_counter += 1
        
        # Get currently visible text and dots
        visible_text = current_phrase[:min(typing_index, len(current_phrase))]
        
        # Calculate dots - now capped at max_dots
        if typing_index > len(current_phrase):
            dots_count = min(typing_index - len(current_phrase), max_dots)
            dots = "." * dots_count
        else:
            dots = ""
        
        # Create the typing effect for the current text
        displayed_text = visible_text[:((i//2) % (len(visible_text) + 1))]
        
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
        
        i += 1
        time.sleep(0.05)
