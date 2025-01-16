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
        
        gradient_text = f"""
        <div style="
            display: flex;
            flex-direction: column;
            gap: 8px;
            ">
            <div style="
                font-size: 24px;
                font-family: monospace;
                padding: 3px 0;
                text-align: left;
                display: flex;
                align-items: center;
                color: rgb(55, 65, 81);
                ">
                <span>{visible_text}{dots}</span>
            </div>
            <div style="
                font-size: 14px;
                font-family: monospace;
                padding: 3px 0;
                text-align: left;
                margin-left: 0;
                display: flex;
                align-items: center;
                ">
                <span style="color: black;">Processing your request{dots}</span>
                <span style="color: rgb(255, 75, 75); margin-left: 4px;">{frame}</span>
            </div>
        </div>
        """
        
        dots_area.markdown(gradient_text, unsafe_allow_html=True)
        time.sleep(0.05)
